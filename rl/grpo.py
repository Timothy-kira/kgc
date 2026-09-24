"""GRPO self-play league training for the TTT+Transformer market controller.

Loop (MiMo-V2.6 / CodeMidas style):
  tasks   = (seed, opponent, seat)  sampled from the league (PFSP over snapshots + fixed + rule adversaries)
  group   = G rollouts of the current policy on the same task (same seed / opponent / seat)
  reward  = win (1 / 0.5 / 0) + margin bonus  (groupwise: extra credit to bigger wins, GAR-like)
  dynamic sampler: drop groups whose rewards are all equal (no relative signal)
  advantage = r - group mean (no std normalisation, as in CodeMidas)
  loss    = PPO-clip on per-(step,product) decisions, prompt(=trajectory)-mean aggregation
            + anchor CE toward the BASE bin (decaying) + value loss + TTT self-supervised loss
League roles: main agent (self-play + league) and a main exploiter (trained only vs. the latest main,
frozen into the pool periodically).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from agent.features import BASE_BIN, N_BINS
from model.net import TTTPolicy
from rl.rollout import rollout

FIXED = ["/home/user/kgc/league/metav4.py", "/home/user/kgc/league/cha22.py", "/home/user/kgc/league/farm2945.py",
         "/home/user/kgc/league/v48.py"]
RULES = ["rule:front_run", "rule:hoarder", "rule:dripper", "rule:noisy"]


def reward_of(me, opp):
    win = 1.0 if me > opp else (0.5 if me == opp else 0.0)
    return win + 0.25 * np.tanh((me - opp) / 15000.0)


class League:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.snapshots = []          # list of weight paths (main + frozen exploiters)
        self.stats = {}              # opponent spec -> [wins, games]  (main agent's perspective)

    def record(self, opp, r):
        s = self.stats.setdefault(opp, [0.0, 0])
        s[0] += 1.0 if r >= 1.0 else (0.5 if r >= 0.5 else 0.0)
        s[1] += 1

    def winrate(self, opp):
        w, n = self.stats.get(opp, [0, 0])
        return (w + 1) / (n + 2)

    def sample_main_opponent(self, latest):
        u = random.random()
        if u < 0.35 or not self.snapshots:
            if u < 0.15 or not self.snapshots:
                return f"ctrl:{latest}@1.0"                       # pure self-play (sampling self)
        if u < 0.70 and self.snapshots:                          # PFSP over past snapshots
            ws = np.array([(1 - self.winrate(f"ctrl:{s}")) ** 2 + 1e-3 for s in self.snapshots])
            return "ctrl:" + self.snapshots[np.random.choice(len(self.snapshots), p=ws / ws.sum())]
        if u < 0.93:
            ws = np.array([(1 - self.winrate(f)) ** 2 + 0.05 for f in FIXED])
            return FIXED[np.random.choice(len(FIXED), p=ws / ws.sum())]
        return random.choice(RULES)


def batchify(trajs):
    T = max(t["T"] for t in trajs)
    B = len(trajs)
    nd = (T + 23) // 24

    def pad(key, shape, dtype):
        out = np.zeros((B,) + shape, dtype)
        for i, t in enumerate(trajs):
            v = t[key]
            out[i, :v.shape[0]] = v
        return torch.tensor(out)
    P = pad("P", (T, 9, trajs[0]["P"].shape[-1]), np.float32)
    G = pad("G", (T, trajs[0]["G"].shape[-1]), np.float32)
    bins = pad("bins", (T, 9), np.int64)
    logp = pad("logp", (T, 9), np.float32)
    mask = pad("mask", (T, 9, N_BINS), bool)
    day_tgt = pad("day_tgt", (nd, trajs[0]["day_tgt"].shape[-1]), np.float32)
    day_mask = pad("day_mask", (nd,), np.float32)
    valid = torch.zeros(B, T, dtype=torch.bool)
    for i, t in enumerate(trajs):
        valid[i, :t["T"]] = True
    return P, G, bins, logp, mask, day_tgt, day_mask, valid, torch.arange(T) // 24


class Learner:
    def __init__(self, name, run_dir, init=None, lr=3e-4):
        self.name = name
        self.net = TTTPolicy()
        if init:
            self.net.load_state_dict(torch.load(init))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.95))
        self.run_dir = run_dir
        self.iter = 0
        self.export()

    @property
    def wpath(self):
        return os.path.join(self.run_dir, f"{self.name}_latest.npz")

    def export(self):
        tmp = self.wpath + ".tmp.npz"
        self.net.export(tmp)
        os.replace(tmp, self.wpath)

    def snapshot(self):
        p = os.path.join(self.run_dir, f"{self.name}_it{self.iter:04d}.npz")
        self.net.export(p)
        torch.save(self.net.state_dict(), p.replace(".npz", ".pt"))
        return p

    def update(self, trajs, adv, anchor_coef, epochs=2, clip=(0.2, 0.28), mb=16):
        stats = {}
        idx = list(range(len(trajs)))
        for ep in range(epochs):
            random.shuffle(idx)
            for s in range(0, len(idx), mb):
                sub = [trajs[i] for i in idx[s:s + mb]]
                A = torch.tensor([adv[i] for i in idx[s:s + mb]], dtype=torch.float32)
                P, G, bins, logp_old, mask, day_tgt, day_mask, valid, day_idx = batchify(sub)
                logits, value, ssl = self.net(P, G, day_idx, day_tgt, day_mask)
                logits = logits.masked_fill(~mask, -1e9)
                logsm = torch.log_softmax(logits, -1)
                lp = logsm.gather(-1, bins.unsqueeze(-1)).squeeze(-1)             # [B,T,9]
                decide = (mask.sum(-1) > 2) & valid.unsqueeze(-1)                  # real choices only
                ratio = torch.exp(lp - logp_old)
                a = A.view(-1, 1, 1)
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - clip[0], 1 + clip[1]) * a)
                n_tok = decide.sum((1, 2)).clamp(min=1)
                pg = -((surr * decide).sum((1, 2)) / n_tok).mean()                # trajectory-mean aggregation
                anchor = -((logsm[..., BASE_BIN] * decide).sum((1, 2)) / n_tok).mean()
                ent = -((logsm.exp() * logsm.clamp(min=-30)).sum(-1) * decide).sum() / decide.sum().clamp(min=1)
                # value head: predict final win prob and money diff from every step
                wins = torch.tensor([1.0 if t["money_me"] > t["money_opp"] else (0.5 if t["money_me"] == t["money_opp"] else 0.0) for t in sub])
                diff = torch.tensor([(t["money_me"] - t["money_opp"]) / 1e4 for t in sub])
                vw = torch.nn.functional.binary_cross_entropy_with_logits(value[..., 0], wins.view(-1, 1).expand_as(value[..., 0]), reduction="none")
                vd = (value[..., 1] - diff.view(-1, 1)) ** 2
                vloss = ((vw + 0.1 * vd) * valid).sum() / valid.sum()
                loss = pg + anchor_coef * anchor + 0.2 * vloss + 0.1 * ssl
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    clipfrac = (((ratio - 1).abs() > 0.2) & decide).sum() / decide.sum().clamp(min=1)
                for k, v in (("pg", pg), ("anchor", anchor), ("ent", ent), ("vloss", vloss), ("ssl", ssl),
                             ("clipfrac", clipfrac)):
                    stats.setdefault(k, []).append(float(v))
        self.iter += 1
        self.export()
        return {k: float(np.mean(v)) for k, v in stats.items()}


def run_groups(pool, learner, tasks, group, temperature):
    jobs = []
    for ti, (seed, opp, seat) in enumerate(tasks):
        for g in range(group):
            jobs.append({"weights": learner.wpath, "temperature": temperature, "opponent": opp, "seed": seed,
                         "seat": seat, "rng_seed": random.getrandbits(31), "tag": ti})
    res = list(pool.imap_unordered(rollout, jobs))
    groups = {}
    for r in res:
        groups.setdefault(r["tag"], []).append(r)
    return groups


def evaluate(pool, wpath, opponents, seeds):
    jobs = [{"weights": wpath, "opponent": o, "seed": s, "seat": seat, "greedy": True, "rng_seed": 0, "tag": o}
            for o in opponents for s in seeds for seat in (0, 1)]
    res = pool.map(rollout, jobs)
    out = {}
    for r in res:
        w = 1.0 if r["money_me"] > r["money_opp"] else (0.5 if r["money_me"] == r["money_opp"] else 0.0)
        out.setdefault(r["tag"], []).append((w, r["money_me"] - r["money_opp"]))
    return {os.path.basename(k): (round(np.mean([x[0] for x in v]), 3), round(np.mean([x[1] for x in v])))
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/grpo1")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--hours", type=float, default=100)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--snapshot_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--exploiter_every", type=int, default=4)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    os.makedirs(a.run, exist_ok=True)
    main_l = Learner("main", a.run, a.init)
    expl = Learner("exploiter", a.run, a.init)
    league = League(a.run)
    log = open(os.path.join(a.run, "log.jsonl"), "a")
    deadline = time.time() + a.hours * 3600
    eval_seeds = list(range(900000, 900004))
    with mp.Pool(a.procs) as pool:
        for it in range(a.iters):
            if time.time() > deadline:
                break
            t0 = time.time()
            train_expl = a.exploiter_every and it % a.exploiter_every == a.exploiter_every - 1
            L = expl if train_expl else main_l
            tasks = []
            for _ in range(a.tasks):
                opp = f"ctrl:{main_l.wpath}" if train_expl else league.sample_main_opponent(main_l.wpath)
                tasks.append((random.randrange(10 ** 6), opp, random.randint(0, 1)))
            groups = run_groups(pool, L, tasks, a.group, a.temperature)
            trajs, adv, kept, rs = [], [], 0, []
            for tag, grp in groups.items():
                r = np.array([reward_of(g["money_me"], g["money_opp"]) for g in grp])
                rs.extend(r.tolist())
                if not train_expl:
                    for g, ri in zip(grp, r):
                        league.record(g["opponent"], ri)
                if r.std() < 1e-6:          # dynamic sampler: no relative signal
                    continue
                kept += 1
                for g, ri in zip(grp, r):
                    if "P" in g:
                        trajs.append(g)
                        adv.append(float(ri - r.mean()))
            anchor = a.anchor * max(0.2, 1.0 - it / 300)
            st = L.update(trajs, adv, anchor) if trajs else {}
            rec = {"it": it, "learner": L.name, "kept_groups": kept, "groups": len(groups), "mean_r": float(np.mean(rs)),
                   "win": float(np.mean([x >= 1.0 for x in rs])), "tie": float(np.mean([0.5 <= x < 1.0 for x in rs])),
                   "sec": round(time.time() - t0, 1), **st,
                   "opps": sorted({os.path.basename(t[1].split('@')[0]) for t in tasks})}
            if not train_expl and main_l.iter % a.snapshot_every == 0:
                league.snapshots.append(main_l.snapshot())
            if train_expl and expl.iter % (a.snapshot_every // 2 or 1) == 0:
                league.snapshots.append(expl.snapshot())      # frozen exploiter joins the pool
            if not train_expl and main_l.iter % a.eval_every == 0:
                rec["eval"] = evaluate(pool, main_l.wpath, FIXED[:2], eval_seeds)
                main_l.snapshot()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
