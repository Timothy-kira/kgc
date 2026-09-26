"""B-track RL (docs/PLAN_RL.md): the pure-Transformer policy (CSA2 decoder + official Engram + CLM heads, init
engram.pt) makes every decision; v4b (cha22 + clamp_sells) is only a training-time teacher / opponent.

python -m rl.tf_rl --ckpt <dir with engram.pt + model_args.json> --pools rl_pools.pkl --vocab opp_vocab.npz
                   --out runs/tf [--hours 11] [--distill_hours 3.5]

Phase 1, distillation (MiMo-V2.6 MOPD analogue with a program teacher): CPU workers play v4b against the opponent
mix and stream the games into a replay buffer (rl/tf_rollout.teacher_game); the learner fits the CLM heads to the
teacher's slot decisions on 240-step crops with the original training loss (model/train_clm.compute_loss: CLM
forward + backward InfoNCE, value, MTP, Engram opponent heads).
Phase 2, GRPO (MiMo / CodeMidas): groups of G games with the same opponent, seed and seat, played by the student at
temperature 1 (batched on the GPU, rl/tf_rollout.run_games); reward = win + 0.25 tanh(diff / 15000); A = r - group
mean (no std); zero-variance groups dropped; MiMo eq. 1 token loss with decoupled clip bounds; + beta KL(pi||pi_ref)
(k3) to the phase-1 policy; + value and Engram opponent auxiliary losses.
EVAL: the same held-out stratified suite as the A track (rl/opp_mix.OppMix.eval_specs, identical jobs), greedy
student vs v4b's cached results (paired win / money differences with SE, per category), plus head-to-head vs v4b.
"""
import argparse
import collections
import json
import multiprocessing
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.clm_batch import collate
from model.clm_policy import CLMPolicy
from model.dsv41 import ModelArgs
from model.opp_data import LAYER_D, LAYER_S, Vocab
from model.seq_batch import obs_feats
from model.train_clm import compute_loss
from infra.fork import warm_pool
from rl import tf_rollout as R
from rl.opp_mix import MIX, OppMix, load_pools

OPP_KEYS = ("eng_s", "eng_d", "opp_tgt", "opp_stock")


def crop(tr, n, rng):
    T = len(tr["prod"])
    if T <= n:
        return tr
    s0 = min(max(rng.randrange(-n + 1, T), 0), T - n)
    s1 = s0 + n
    off = tr["dec_off"]
    a0, a1 = int(off[s0]), int(off[s1])
    out = {k: tr[k][s0:s1] for k in ("prod", "glob", "tiles", "units") + OPP_KEYS}
    out.update(dec_slot=tr["dec_slot"][a0:a1], dec_cand=tr["dec_cand"][a0:a1], dec_off=off[s0:s1 + 1] - a0)
    for k in ("win", "diff", "score"):
        out[k] = tr[k]
    return out


def load_net(ckpt, weights, device):
    d = json.load(open(os.path.join(ckpt, "model_args.json")))
    args = ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in ModelArgs.__dataclass_fields__})
    net = CLMPolicy(args).to(device)
    sd = torch.load(weights or os.path.join(ckpt, "engram.pt"), map_location=device)
    print("init", net.load_state_dict(sd, strict=False), flush=True)
    return net, d


def reward_of(t):
    return t["win"] + 0.25 * float(np.tanh(t["diff"] * 1e4 / 15000.0))


def logprobs(net, trajs, device, dtype):
    """Per-decision log-probs of the recorded decisions (same CLM logits as sampling) + value + opponent heads."""
    b = collate(trajs, device=device, loser_w=1.0)
    net.engram_ids, net.engram_mask = {LAYER_S: b["eng_s"], LAYER_D: b["eng_d"]}, b["eng_mask"]
    lu, lm, value, _, aux = net.forward_train(b["ids"], obs_feats(b, dtype), b["otype"], b["dec_pos"], b["dec_slot"],
                                              b["dec_desc"], b["tgt_pos"], b["tgt_kind"], b["val_pos"], None, None,
                                              None, b["opp_pos"])
    k, c = b["tgt_kind"], b["tgt_cand"]
    lp = torch.zeros(len(c), device=device)
    if (k == 0).any():
        lp[k == 0] = torch.log_softmax(lu.float(), -1).gather(1, c[k == 0][:, None]).squeeze(1)
    if (k == 1).any():
        lp[k == 1] = torch.log_softmax(lm.float(), -1).gather(1, c[k == 1][:, None]).squeeze(1)
    return lp, b["tgt_pos"][:, 0], value, b, aux


def evaluate(net, device, workers, suite, base, n_h2h, t_h, tag, max_steps=None):
    specs = [s for _, s in suite] + [("h2h", "v4b", 1_990_000_000 + i, i % 2, None, ("v4b",)) for i in range(n_h2h)]
    net.eval()
    with torch.no_grad():
        trajs = R.run_games(net, device, workers, specs, temperature=0.0, max_steps=max_steps)
    w = lambda d: 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    pol = {j: (s[0], t["money_me"] - t["money_opp"]) for (j, s), t in zip(suite, trajs[:len(suite)])}
    h2h = [t["money_me"] - t["money_opp"] for t in trajs[len(suite):]]
    js = sorted(pol)
    wp, wb = np.array([w(pol[j][1]) for j in js]), np.array([w(base[j][1]) for j in js])
    dp = np.array([pol[j][1] - base[j][1] for j in js], np.float64)
    by = collections.defaultdict(lambda: [[], []])
    for j in js:
        by[pol[j][0]][0].append(w(pol[j][1]))
        by[pol[j][0]][1].append(w(base[j][1]))
    by = {k: [round(float(np.mean(a)), 3), round(float(np.mean(b)), 3)] for k, (a, b) in by.items()}
    se = float((wp - wb).std() / np.sqrt(len(js)))
    return {"tag": tag, "t_h": round(t_h, 2), "n": len(js), "win": round(float(wp.mean()), 3),
            "win_base": round(float(wb.mean()), 3), "dwin": round(float((wp - wb).mean()), 3), "dwin_se": round(se, 3),
            "money": round(float(np.mean([t["money_me"] for t in trajs[:len(suite)]]))),
            "ddiff": round(float(dp.mean())), "ddiff_se": round(float(dp.std() / np.sqrt(len(js)))), "by_kind": by,
            "h2h_win": round(float(np.mean([w(d) for d in h2h])), 3) if h2h else None,
            "h2h_diff": round(float(np.mean(h2h))) if h2h else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--init", default=None, help="weights to start from (default <ckpt>/engram.pt)")
    ap.add_argument("--pools", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="runs/tf")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--distill_hours", type=float, default=3.5)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--buf", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rl_lr", type=float, default=1e-5)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--clip_pos", type=float, nargs=2, default=(0.8, 1.28))
    ap.add_argument("--clip_neg", type=float, nargs=2, default=(0.72, 1.2))
    ap.add_argument("--eval_min", type=float, default=60.0)
    ap.add_argument("--h2h", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="truncate games (smoke tests)")
    ap.add_argument("--eval_n", type=int, default=0, help="limit the eval suite (smoke tests)")
    ap.add_argument("--log_sec", type=float, default=300.0)
    ap.add_argument("--eval_pools", default=None, help="pools file for the eval suite (default --pools)")
    ap.add_argument("--demo_frac", type=float, default=0.0, help="phase 1: share of replayed high-rated winners")
    ap.add_argument("--grpo_mix", default="", help="e.g. near=0.25,loss=0.10,top=0.10,live=0.20,self=0.10")
    ap.add_argument("--sp_frac", type=float, default=0.0, help="phase 2: share of self-play games (PFSP snapshots)")
    ap.add_argument("--sp_min_win", type=float, default=0.2, help="self-play only once the last EVAL win >= this")
    ap.add_argument("--snap_every", type=int, default=4, help="GRPO iterations between self-play snapshots")
    ap.add_argument("--snap_keep", type=int, default=4)
    ap.add_argument("--skip_first_eval", type=int, default=0, help="1: no eval before training (baseline known)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    dtype = amp_dtype if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    procs = a.procs or max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(1 if use_amp else 2)          # the GPU process must not oversubscribe the CPU workers
    os.makedirs(a.out, exist_ok=True)
    vocab = Vocab(a.vocab)
    R.W["vocab"] = vocab
    train_pools = load_pools(a.pools, "train")
    R.W["mix"] = OppMix(train_pools, MIX)
    R.W["demo"], R.W["demo_frac"] = train_pools.get("demo"), a.demo_frac
    suite = OppMix(load_pools(a.eval_pools or a.pools, "eval"), seed_base=1_999_000_000).eval_specs()
    gmix = dict(MIX)
    for kv in filter(None, a.grpo_mix.split(",")):
        k, v = kv.split("=")
        gmix[k] = float(v)
    grpo_mix = OppMix(train_pools, gmix)
    emit_mix = {"pools": {k: len(v) for k, v in train_pools.items()}, "demo_frac": a.demo_frac, "grpo_mix": grpo_mix.mix,
                "sp_frac": a.sp_frac}
    if a.eval_n:
        suite = suite[::max(1, len(suite) // a.eval_n)][:a.eval_n]
    latest = os.path.join(a.out, "tf_latest.pt")
    state_p = os.path.join(a.out, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"phase": 1, "elapsed_h": 0.0, "n_eval": 0}
    net, margs = load_net(a.ckpt, latest if os.path.exists(latest) else a.init, device)
    json.dump(margs, open(os.path.join(a.out, "model_args.json"), "w"))
    tab = [p for n, p in net.named_parameters() if n.startswith("engrams.") and n.endswith("embed.weight")]
    tab_ids = {id(p) for p in tab}
    groups = [{"params": [p for p in net.parameters() if id(p) not in tab_ids], "mult": 1.0},
              {"params": tab, "mult": 5.0, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(kind, rec):
        print(kind + " " + json.dumps(rec), flush=True)
        log.write(json.dumps({kind: rec}) + "\n")
        log.flush()

    t_start = time.time() - state["elapsed_h"] * 3600
    deadline = t_start + a.hours * 3600
    t_h = lambda: (time.time() - t_start) / 3600
    base_p = os.path.join(a.out, "base_eval.json")
    if os.path.exists(base_p):
        base = {int(k): tuple(v) for k, v in json.load(open(base_p)).items()}
    else:
        with warm_pool(procs) as bp:
            base = {j: (kind, d) for j, kind, d in bp.imap_unordered(R.base_game, suite)}
        json.dump(base, open(base_p, "w"))
    emit("MIX", emit_mix)
    emit("BASE", {"n": len(base), "win": round(float(np.mean([1.0 if d > 0 else 0.5 if d == 0 else 0.0 for _, d in base.values()])), 3)})
    workers = R.Workers(procs, vocab)
    best = state.get("best_key")
    last_win = [state.get("last_win", 0.0)]
    last_eval = time.time() if a.skip_first_eval else -1e9
    rng = random.Random(int(time.time()))
    gen = {"pool": None, "it": None}

    def gen_start():
        gen["pool"] = warm_pool(procs, maxtasksperchild=50)
        gen["it"] = gen["pool"].imap_unordered(R.teacher_game, (rng.randrange(10 ** 8) for _ in iter(int, 1)))

    def gen_stop():
        if gen["pool"] is not None:
            gen["pool"].terminate()
            gen["pool"].join()
            gen["pool"] = gen["it"] = None

    def maybe_eval(tag):
        nonlocal last_eval, best
        if time.time() - last_eval < a.eval_min * 60:
            return
        running = gen["pool"] is not None
        gen_stop()                                     # the eval rollouts get every CPU core
        rec = evaluate(net, device, workers, suite, base, a.h2h, t_h(), tag, a.max_steps or None)
        rec["eval"] = state["n_eval"]
        emit("EVAL", rec)
        last_win[0] = rec["win"]
        key = [rec["dwin"], rec["ddiff"]]
        torch.save(net.state_dict(), latest)
        if best is None or key > best:
            best = key
            torch.save(net.state_dict(), os.path.join(a.out, "tf_best.pt"))
            json.dump(rec, open(os.path.join(a.out, "best.json"), "w"))
        state.update(n_eval=state["n_eval"] + 1, best_key=best, elapsed_h=t_h(), last_win=last_win[0])
        json.dump(state, open(state_p, "w"))
        last_eval = time.time()
        net.train()
        if running:
            gen_start()

    # ------------------------------------------------------------------ phase 1: distillation
    if state["phase"] == 1 and a.distill_hours > 0:
        buf = collections.deque(maxlen=a.buf)
        gen_start()
        coefs = {"value": 0.2, "mtp": 0.3, "index": 0.1, "bwd": 1.0, "policy": 1.0, "opp": 0.5}
        step, n_games, stats, t_log = 0, 0, collections.defaultdict(list), time.time()
        p1_end = t_start + a.distill_hours * 3600
        while time.time() < min(p1_end, deadline):
            maybe_eval("distill")
            while True:
                if gen["it"] is None:
                    break
                try:
                    tr = gen["it"].next(timeout=0.0 if len(buf) >= 2 * a.batch else 60)
                except multiprocessing.TimeoutError:
                    break
                buf.append(tr)
                n_games += 1
                if len(buf) < 2 * a.batch:
                    continue
                if n_games % 8 == 0:
                    break
            if len(buf) < 2 * a.batch:
                continue
            batch = [crop(buf[rng.randrange(len(buf))], a.crop, rng) for _ in range(a.batch)]
            b = collate(batch, device=device, loser_w=1.0, hist_drop=0.5)
            lr = a.lr * min(1.0, (step + 1) / 100)
            for g in opt.param_groups:
                g["lr"] = lr * g["mult"]
            net.train()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, st = compute_loss(net, b, dtype, coefs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            net.engram_ids = None
            step += 1
            for k in ("ce", "acc_unit", "acc_market", "opp_acc", "vwin"):
                stats[k].append(st[k])
            if time.time() - t_log > a.log_sec:
                emit("DISTILL", {"step": step, "games": n_games, "buf": len(buf), "t_h": round(t_h(), 2),
                                 "win_teacher": round(float(np.mean([t["win"] for t in buf])), 3),
                                 **{k: round(float(np.mean(v)), 4) for k, v in stats.items()}})
                stats, t_log = collections.defaultdict(list), time.time()
                torch.save(net.state_dict(), latest)
                state["elapsed_h"] = t_h()
                json.dump(state, open(state_p, "w"))
        gen_stop()
        last_eval = -1e9
        maybe_eval("distill_end")
        torch.save(net.state_dict(), os.path.join(a.out, "tf_phase1.pt"))
        state["phase"] = 2
        json.dump(state, open(state_p, "w"))
    gen_stop()

    # ------------------------------------------------------------------ phase 2: GRPO
    ref_path = os.path.join(a.out, "tf_phase1.pt")
    ref, _ = load_net(a.ckpt, ref_path if os.path.exists(ref_path) else (a.init if not os.path.exists(latest) else latest), device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    for g in opt.param_groups:
        g["lr"] = a.rl_lr * g["mult"]
    it = 0
    snaps, sp_win, opp_net = [], {}, [None]
    while time.time() < deadline - 0.1 * 3600:
        maybe_eval("grpo")
        t0 = time.time()
        if it % a.snap_every == 0:
            snaps.append(({k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}, f"it{it}"))
            snaps[:] = snaps[-a.snap_keep:]
        sp_on = a.sp_frac > 0 and last_win[0] >= a.sp_min_win
        si = None
        if sp_on:
            w = [(1.0 - sp_win.get(tag, 0.5)) ** 2 + 0.05 for _, tag in snaps]
            si = rng.choices(range(len(snaps)), weights=w)[0]
            if opp_net[0] is None:
                opp_net[0], _ = load_net(a.ckpt, ref_path if os.path.exists(ref_path) else None, device)
                opp_net[0].eval()
            opp_net[0].load_state_dict(snaps[si][0])
        specs, gid = [], []
        for k in range(a.tasks):
            j = rng.randrange(10 ** 8)
            if sp_on and rng.random() < a.sp_frac:
                s = ("sp", snaps[si][1], 30000 + j, j % 2, None, ("self", snaps[si][1]))
            else:
                s = grpo_mix.spec(j)
            specs += [s] * a.group
            gid += [k] * a.group
        net.eval()
        with torch.no_grad():
            trajs = R.run_games(net, device, workers, specs, temperature=1.0, max_steps=a.max_steps or None,
                                opp_net=opp_net[0] if sp_on else None)
        for t in trajs:
            if t["kind"] == "sp":
                tag = t["opponent"]
                sp_win[tag] = 0.9 * sp_win.get(tag, 0.5) + 0.1 * t["win"]
        r = np.array([reward_of(t) for t in trajs])
        adv = np.zeros(len(trajs))
        kept = 0
        for k in range(a.tasks):
            idx = [i for i, g in enumerate(gid) if g == k]
            if r[idx].std() < 1e-8:
                continue
            kept += 1
            adv[idx] = r[idx] - r[idx].mean()
        use = [i for i in range(len(trajs)) if adv[i] != 0.0]
        random.shuffle(use)
        st = collections.defaultdict(list)
        net.train()
        for i in use:
            t = trajs[i]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                lp, tid, value, b, _ = logprobs(net, [t], device, dtype)
                ol = net.last_opp
                with torch.no_grad():
                    lp_ref, _, _, _, _ = logprobs(ref, [t], device, dtype)
            old = torch.from_numpy(t["lp"]).to(device)
            ratio = torch.exp(lp - old).detach()
            A = float(adv[i])
            lo, hi = (a.clip_pos if A >= 0 else a.clip_neg)
            M = ((ratio >= lo) & (ratio <= hi)).float()
            pg = -(ratio * M * A * lp).mean()
            kl = (torch.exp(lp_ref - lp) - (lp_ref - lp) - 1).mean()
            vloss = F.binary_cross_entropy_with_logits(value[:, 0].float(), torch.full_like(value[:, 0].float(), t["win"])) + \
                0.1 * F.mse_loss(value[:, 1].float(), torch.full_like(value[:, 1].float(), t["diff"]))
            opp_ce = F.cross_entropy(ol[0].float().flatten(0, 2), b["opp_tgt"].flatten()) if ol is not None else torch.zeros((), device=device)
            loss = pg + a.beta * kl + 0.1 * vloss + 0.1 * opp_ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            net.engram_ids = ref.engram_ids = None
            for k2, v in (("pg", pg), ("kl", kl), ("vloss", vloss), ("clipfrac", 1 - M.mean()), ("ratio", ratio.mean())):
                st[k2].append(float(v))
        by = collections.defaultdict(list)
        for t in trajs:
            by[t["kind"]].append(t["win"])
        emit("ITER", {"it": it, "games": len(trajs), "kept_groups": kept, "t_h": round(t_h(), 2),
                      "win": round(float(np.mean([t["win"] for t in trajs])), 3),
                      "money": round(float(np.mean([t["money_me"] for t in trajs]))),
                      "mean_r": round(float(r.mean()), 4), "sec": round(time.time() - t0),
                      **{k: round(float(np.mean(v)), 5) for k, v in st.items()},
                      "win_by_kind": {k: round(float(np.mean(v)), 3) for k, v in by.items()},
                      "sp": {"on": sp_on, "snap": snaps[si][1] if si is not None else None,
                             "snap_win": {k: round(v, 3) for k, v in sp_win.items()}}})
        torch.save(net.state_dict(), latest)
        state["elapsed_h"] = t_h()
        json.dump(state, open(state_p, "w"))
        it += 1
    if it > 0:                                         # nothing new to evaluate without GRPO iterations
        last_eval = -1e9
        maybe_eval("final")
    workers.close()


if __name__ == "__main__":
    main()
