"""A-track RL (docs/PLAN_RL.md): key-step fork policy improvement for the v5 residual on top of v4b.

python -m rl.v5_rl --pools rl_pools.pkl --vocab opp_vocab.npz --out runs/v5 [--hours 11] [--procs 4]

Actors (forked CPU workers under SCHED_IDLE, infra/fork.py): a task plays one trunk game with the current greedy
policy against an opponent from the mix (rl/opp_mix.py). At M random decision steps it forks a COW snapshot of the
live game (env + cha22 + agent state) into one branch per alternative action of one randomly chosen head (the other
heads keep their greedy action); every branch continues greedily to the end. The trunk is the greedy branch of all
its fork points, so each sample carries the exact return of every allowed action of that head (a GRPO group that
enumerates the head's actions): A(a) = r(a) - mean_group r, r = 1{diff>0} + 0.5*1{diff=0} + diff/20000.

Learner: maximise sum_a pi(a|s) A(s,a) (exact expectation over the group, so no importance ratio is needed)
- beta KL(pi || pi_prior) + eta H(pi). Engram tables at 5x lr without weight decay (Engram paper).

--obj adv (advantage regression): the head's logits are read as advantages in units of 0.01 reward (200 money):
Huber(lg[a] - lg[greedy], (r(a) - r(greedy)) / 0.01) on the enumerated actions; acting greedily after adding
--margin to the follow logit, i.e. deviate only where the predicted gain exceeds the margin. Unlike the policy
gradient (whose signal on an action vanishes with its probability), this keeps learning where follow dominates.
Every update's samples are saved (samples_NNNN.pkl) for offline re-use.

Monitoring
  ITER  per update: samples, samples/min, headroom = mean max_a r(a) - r(greedy), mean |A|, share of samples whose
        best action is not the greedy one, greedy deviation rate (heads whose greedy action is not follow), entropy,
        KL to the prior, loss, headroom by category.
  EVAL  every --eval_min minutes: greedy policy vs v4b on a fixed held-out stratified suite (paired: same opponent
        tape / league agent, seed and seat; tapes from held-out episodes), win-rate and money differences with paired
        SE, per-category win rates, plus head-to-head games vs v4b. PROMOTE when dwin > 2 SE and no category loses
        more than 2 pp (then re-check locally on >= 300 games before packing v5).
"""
import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool
from model.opp_data import Vocab
from rl.opp_mix import MIX, OppMix, load_pools, v4b
from rl.v5_agent import CTRL, DECIDE_EVERY, FOLLOW, N_ACT, V5Agent
from rl.v5_policy import PolicyRunner, RLPolicy, batch_inputs

G = {}                                   # worker globals (inherited through fork)
LADDER = {"near": 0.55, "loss": 0.15, "top": 0.10, "live": 0.15, "self": 0.05}


def reward(diff):
    return float(diff > 0) + 0.5 * float(diff == 0) + diff / 20000.0


def _net(path):
    """Worker-side policy, reloaded when the weights file changes."""
    st = os.stat(path).st_mtime_ns if path and os.path.exists(path) else None
    if G.get("net_key") != (path, st):
        net = RLPolicy(G["vocab"].sizes, ens=G.get("ens", 1))
        if st is not None:
            net.load_state_dict(torch.load(path, map_location="cpu"))
        G["runner"], G["net_key"] = PolicyRunner(net, G.get("margin", 0.0), G.get("lcb_c", 0.0)), (path, st)
    return G["runner"]


def _play(env, me, opp, seat):
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    return env.money[seat] - env.money[1 - seat]


def train_task(args):
    j, weights, n_forks = args
    best_effort()
    torch.set_num_threads(1)
    runner = _net(weights)
    kind, name, seed, seat, cfg, how = G["mix"].spec(j)
    opp = OppMix.make(how)
    me = V5Agent(G["vocab"], policy_fn=runner.greedy)
    env = FarmEnv(seed, cfg)
    rng = random.Random(j)
    fork_at = set(DECIDE_EVERY * d for d in rng.sample(range(2, 120), n_forks))
    samples, t0 = [], time.time()
    while not env.done:
        t = env.step_count
        if t in fork_at:
            inp = me.inputs(env.obs(seat))
            greedy, _ = runner.greedy(inp)
            al = inp["allowed"]
            heads = [k for k in range(len(CTRL)) if any(al[k, a] and a != greedy[k] for a in range(N_ACT))]
            if heads:
                k = rng.choice(heads)
                alts = [a for a in range(N_ACT) if a != greedy[k] and al[k, a]]

                def branch(a, k=k, greedy=greedy):
                    acts = list(greedy)
                    acts[k] = a
                    me.override = acts
                    return _play(env, me, opp, seat)

                res = fork_branches(alts, branch, max_parallel=1)
                samples.append(dict(inp=inp, head=k, greedy=greedy[k], alts=alts, res=res, t=t))
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    final = env.money[seat] - env.money[1 - seat]
    out = []
    for s in samples:
        r = np.full(N_ACT, np.nan, np.float32)
        r[s["greedy"]] = reward(final)
        for a, v in zip(s["alts"], s["res"]):
            if not isinstance(v, Exception):
                r[a] = reward(v)
        out.append(dict(inp=s["inp"], head=s["head"], greedy=s["greedy"], r=r, t=s["t"], kind=kind))
    return dict(job=j, kind=kind, final=final, samples=out, sec=time.time() - t0)


def eval_task(args):
    """(tag, j, spec, weights) -> (tag, j, kind, diff): tag 'pol' / 'base' on the suite, 'h2h' vs v4b."""
    tag, j, spec, weights = args
    best_effort()
    torch.set_num_threads(1)
    kind, name, seed, seat, cfg, how = spec
    if tag == "base":
        me = v4b()
    else:
        me = V5Agent(G["vocab"], policy_fn=_net(weights).greedy)
    opp = v4b() if tag == "h2h" else OppMix.make(how)
    return tag, j, kind, _play(FarmEnv(seed, cfg), me, opp, seat)


def evaluate(pool, suite, base, weights, n_h2h, t_h):
    tasks = [("pol", j, spec, weights) for j, spec in suite]
    tasks += [("h2h", 2 * 10 ** 6 + i, ("h2h", "v4b", 1_990_000_000 + i, i % 2, None, None), weights) for i in range(n_h2h)]
    pol, h2h = {}, []
    for tag, j, kind, d in pool.imap_unordered(eval_task, tasks):
        if tag == "pol":
            pol[j] = (kind, d)
        else:
            h2h.append(d)
    js = sorted(pol)
    w = lambda d: 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    wp, wb = np.array([w(pol[j][1]) for j in js]), np.array([w(base[j][1]) for j in js])
    dp = np.array([pol[j][1] - base[j][1] for j in js], np.float64)
    by = {}
    for j in js:
        by.setdefault(pol[j][0], [[], []])
        by[pol[j][0]][0].append(w(pol[j][1]))
        by[pol[j][0]][1].append(w(base[j][1]))
    by = {k: [round(float(np.mean(a)), 3), round(float(np.mean(b)), 3)] for k, (a, b) in by.items()}
    se = float((wp - wb).std() / np.sqrt(len(js)))
    dwin = float((wp - wb).mean())
    worst = min(a - b for a, b in by.values())
    # ladder-weighted paired win difference (category shares of our ladder games, docs/PLAN_RL.md)
    cats = {}
    for j in js:
        cats.setdefault(pol[j][0], []).append(w(pol[j][1]) - w(base[j][1]))
    W = {c: LADDER.get(c, 0.0) for c in cats}
    tot = sum(W.values()) or 1.0
    wd = sum(W[c] / tot * float(np.mean(v)) for c, v in cats.items())
    wse = float(np.sqrt(sum((W[c] / tot) ** 2 * float(np.var(v)) / len(v) for c, v in cats.items())))
    rec = {"t_h": round(t_h, 2), "n": len(js), "win": round(float(wp.mean()), 3), "win_base": round(float(wb.mean()), 3),
           "dwin": round(dwin, 3), "dwin_se": round(se, 3), "ddiff": round(float(dp.mean())),
           "ddiff_se": round(float(dp.std() / np.sqrt(len(js)))), "changed": int((dp != 0).sum()), "by_kind": by,
           "h2h_win": round(float(np.mean([w(d) for d in h2h])), 3) if h2h else None,
           "h2h_diff": round(float(np.mean(h2h))) if h2h else None,
           "wdwin": round(wd, 4), "wdwin_se": round(wse, 4),
           "promote": bool(wd > 2 * wse and wd > 0 and worst >= -0.02)}
    return rec


def update(net, prior, opt, samples, a):
    X = batch_inputs([s["inp"] for s in samples])
    R = torch.from_numpy(np.stack([s["r"] for s in samples]))
    valid = ~torch.isnan(R)
    Rz = torch.where(valid, R, torch.zeros_like(R))
    mean = Rz.sum(1, keepdim=True) / valid.sum(1, keepdim=True)
    A = torch.where(valid, R - mean, torch.zeros_like(R))
    k = torch.tensor([s["head"] for s in samples])
    G_idx = torch.tensor([s["greedy"] for s in samples])
    boot = torch.poisson(torch.ones(len(samples), getattr(net, "ens", 1)))      # bootstrap (Poisson(1)) per member
    with torch.no_grad():
        lp0 = torch.log_softmax(prior(*X), -1)
    n, st = len(samples), {}
    for _ in range(a.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, a.mb):
            idx = perm[i:i + a.mb]
            xb = tuple(x[idx] for x in X)
            raw_all = net.forward_all(*xb)                                    # [n,ens,5,3]
            raw = raw_all.mean(1)
            lp = torch.log_softmax(raw, -1)
            p = lp.exp()
            al = xb[4]
            ii = torch.arange(len(idx))
            pk = p[ii, k[idx]]
            kl = torch.where(al, p * (lp - lp0[idx]), torch.zeros_like(p)).sum(-1).mean()
            ent = -torch.where(al, p * lp, torch.zeros_like(p)).sum(-1).mean()
            if a.obj == "adv":
                E = raw_all.size(1)
                lgk = raw_all[ii, :, k[idx]]                                  # [n,ens,3]
                gi = G_idx[idx]
                pred = lgk - lgk.gather(2, gi.view(-1, 1, 1).expand(-1, E, 1))
                tgt = ((R[idx] - R[idx].gather(1, gi[:, None])) / 0.01).clamp(-50, 50)
                tgt = torch.nan_to_num(tgt).unsqueeze(1).expand_as(pred)
                m = (valid[idx] & (torch.arange(3)[None] != gi[:, None])).unsqueeze(1).float()
                wb = boot[idx].unsqueeze(-1)                                  # bootstrap weights per member
                el = F.huber_loss(pred, tgt, delta=5.0, reduction="none")
                den = (m * wb).sum()
                pg = (el * m * wb).sum() / den if den > 0 else raw.sum() * 0
                loss = pg
            else:
                pg = -(pk * A[idx]).sum(-1).mean()
                loss = pg + a.beta * kl - a.eta * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            for kk, v in (("pg", pg), ("kl", kl), ("ent", ent), ("loss", loss)):
                st.setdefault(kk, []).append(float(v))
    with torch.no_grad():
        la = net.forward_all(*X)
        if a.obj == "adv":
            gain = la - la[..., FOLLOW:FOLLOW + 1]
            score = gain.mean(1) - (a.lcb_c * gain.std(1) if la.size(1) > 1 else 0)
            score[..., FOLLOW] = a.margin
            lg = score.masked_fill(~X[4], -1e9)
        else:
            lg = la.mean(1)
        choice = X[4].sum(-1) > 1                                  # heads with a real choice
        dev = float((lg.argmax(-1) != FOLLOW)[choice].float().mean()) if choice.any() else 0.0
    greedy_r = torch.stack([R[i, s["greedy"]] for i, s in enumerate(samples)])
    best = torch.where(valid, R, torch.full_like(R, -1e9)).max(1)
    head = (best.values - greedy_r)
    by = {}
    for s, h in zip(samples, head.tolist()):
        by.setdefault(s["kind"], []).append(h)
    return {"headroom": round(float(head.mean()), 5), "abs_adv": round(float(A[valid].abs().mean()), 5),
            "better_than_greedy": round(float((head > 0).float().mean()), 3),
            "greedy_dev": round(dev, 4), **{kk: round(float(np.mean(v)), 5) for kk, v in st.items()},
            "headroom_by_kind": {kk: round(float(np.mean(v)), 4) for kk, v in by.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="runs/v5")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--forks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--eval_min", type=float, default=40.0)
    ap.add_argument("--h2h", type=int, default=20)
    ap.add_argument("--mix", default="")
    ap.add_argument("--obj", choices=["pg", "adv"], default="pg")
    ap.add_argument("--margin", type=float, default=1.0, help="adv mode: follow-logit margin when acting")
    ap.add_argument("--ens", type=int, default=1, help="bootstrap ensemble heads (adv mode)")
    ap.add_argument("--lcb_c", type=float, default=0.0, help="act only if mean - c*std of the ensemble gain > margin")
    ap.add_argument("--ema", type=float, default=0.0, help="act / evaluate with EMA weights (decay per update)")
    ap.add_argument("--eval_pools", default="", help="whole-DB pools: EVAL on the ~310-game fresh suite (tools/v5_recheck)")
    ap.add_argument("--learner_threads", type=int, default=1)
    ap.add_argument("--save_frac", type=float, default=1.0, help="share of update batches saved (compressed)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    if a.procs <= 0:                                          # TPU VM host: 224 cores, 377 GB; ~1.3 GB per worker
        mem_gb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2 ** 20
        a.procs = max(2, min((os.cpu_count() or 4) - 16, int(mem_gb / 1.3)))
    print(json.dumps({"procs": a.procs, "cpus": os.cpu_count()}), flush=True)
    G["vocab"] = Vocab(a.vocab)
    G["margin"] = a.margin if a.obj == "adv" else 0.0
    G["ens"], G["lcb_c"] = a.ens, a.lcb_c
    mix = dict(MIX)
    for kv in filter(None, a.mix.split(",")):
        k, v = kv.split("=")
        mix[k] = float(v)
    G["mix"] = OppMix(load_pools(a.pools, "train"), mix)
    if a.eval_pools:
        from tools.v5_recheck import COUNTS
        ev = load_pools(a.eval_pools, "eval")
        counts = {k: (min(v, len(ev.get(k, []))) if k not in ("live", "self") else v) for k, v in COUNTS.items()}
        suite = OppMix(ev, seed_base=1_700_000_000).eval_specs(counts, start=3 * 10 ** 6)
    else:
        suite = OppMix(load_pools(a.pools, "eval"), seed_base=1_999_000_000).eval_specs()
    torch.manual_seed(0)
    net = RLPolicy(G["vocab"].sizes, ens=a.ens)
    latest = os.path.join(a.out, "policy_latest.pt")
    if os.path.exists(latest):
        net.load_state_dict(torch.load(latest))
        print("resumed", latest, flush=True)
    ema = RLPolicy(G["vocab"].sizes, ens=a.ens).eval()
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    acting = ema if a.ema > 0 else net                        # weights the actors / EVAL use
    torch.manual_seed(0)
    prior = RLPolicy(G["vocab"].sizes, ens=a.ens).eval()                   # = the initial policy (greedy = v4b)
    for p in prior.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.param_groups(a.lr), betas=(0.9, 0.99))
    torch.save(acting.state_dict(), latest)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t_start = time.time()
    deadline = t_start + a.hours * 3600
    with warm_pool(a.procs, maxtasksperchild=50) as pool:
        torch.set_num_threads(a.learner_threads)
        base_path = os.path.join(a.out, "base_eval.json")
        if os.path.exists(base_path):
            base = {int(k): tuple(v) for k, v in json.load(open(base_path)).items()}
        else:
            base = {}
            for tag, j, kind, d in pool.imap_unordered(eval_task, [("base", j, spec, None) for j, spec in suite]):
                base[j] = (kind, d)
            json.dump(base, open(base_path, "w"))
        wb = np.mean([1.0 if d > 0 else 0.5 if d == 0 else 0.0 for _, d in base.values()])
        print(json.dumps({"BASE": {"n": len(base), "win": round(float(wb), 3)}}), flush=True)
        n_eval, last_eval, best = 0, -1e9, None
        it, j_next = 0, random.randrange(10 ** 7) * 10
        buf, n_tasks, t_it = [], 0, time.time()
        pending = collections.deque()             # bounded queue, so EVAL tasks never wait behind a long backlog
        while time.time() < deadline:
            if time.time() - last_eval > a.eval_min * 60:
                snap = os.path.join(a.out, f"policy_eval{n_eval:03d}.pt")
                torch.save(acting.state_dict(), snap)
                rec = evaluate(pool, suite, base, snap, a.h2h, (time.time() - t_start) / 3600)
                rec.update(eval=n_eval, it=it, weights=os.path.basename(snap))
                print("EVAL " + json.dumps(rec), flush=True)
                log.write(json.dumps({"EVAL": rec}) + "\n")
                log.flush()
                key = (rec["dwin"], rec["ddiff"])
                if best is None or key > best[0]:
                    best = (key, snap)
                    json.dump({"best": os.path.basename(snap), **rec}, open(os.path.join(a.out, "best.json"), "w"))
                n_eval, last_eval = n_eval + 1, time.time()
            while len(pending) < 2 * a.procs:
                pending.append(pool.apply_async(train_task, ((j_next, latest, a.forks),)))
                j_next += 1
            try:
                r = pending.popleft().get()
            except Exception as e:                # one broken game must not stop training
                print(json.dumps({"task_error": repr(e)[:300]}), flush=True)
                continue
            n_tasks += 1
            buf += r["samples"]
            if len(buf) < a.batch:
                continue
            if random.random() < a.save_frac:                  # compressed: feats fp16, hash rows int32
                import pickle
                comp = [dict(s_, inp={k: (v.astype(np.float16) if k == "feats" else v.astype(np.int32) if k.startswith("rows")
                                          else v) for k, v in s_["inp"].items()}) for s_ in buf]
                pickle.dump(comp, open(os.path.join(a.out, f"samples_{it:04d}.pkl"), "wb"))
            st = update(net, prior, opt, buf, a)
            if a.ema > 0:
                with torch.no_grad():
                    for pe, p_ in zip(ema.parameters(), net.parameters()):
                        pe.mul_(a.ema).add_(p_, alpha=1 - a.ema)
            torch.save(acting.state_dict(), latest + ".tmp")
            os.replace(latest + ".tmp", latest)
            dt = time.time() - t_it
            rec = {"it": it, "samples": len(buf), "tasks": n_tasks, "spm": round(len(buf) / dt * 60, 1),
                   "t_h": round((time.time() - t_start) / 3600, 2), **st}
            print("ITER " + json.dumps(rec), flush=True)
            log.write(json.dumps({"ITER": rec}) + "\n")
            log.flush()
            it, buf, n_tasks, t_it = it + 1, [], 0, time.time()
        pool.terminate()


if __name__ == "__main__":
    main()
