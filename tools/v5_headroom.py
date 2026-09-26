"""Headroom of the v5 residual (docs/PLAN_RL.md): at a random decision step, how much does a single hold/dump
deviation from v4b change the final money difference?

python -m tools.v5_headroom <out.jsonl> <n_groups> [procs=4] [db_dir] [vocab]
Each group: an opponent from the RL mix, v4b (all follow) up to a random decision step, then forks: all-follow plus
every allowed single-product deviation (hold or dump for one product, the rest follow); every branch continues
all-follow to the end. Prints per-action mean gain vs follow and the per-group best gain (upper bound of one
decision's value).
"""
import json
import os
import random
import sys

import numpy as np

import tools.route_data as RD
from agent.loader import load_agent
from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool
from model.opp_data import Vocab
from rl.v5_agent import CTRL, DECIDE_EVERY, DUMP, FOLLOW, HOLD, V5Agent

VOCAB = None


def run_group(j):
    best_effort()
    kind, opp_name, seed, seat, cfg, tape = RD.job_spec(j)
    opp = tape if tape is not None else load_agent(opp_name)
    me = V5Agent(VOCAB)
    env = FarmEnv(seed, cfg)
    t_dec = DECIDE_EVERY * random.Random(j).randrange(8, 118)
    while env.step_count < t_dec and not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    obs = env.obs(seat)
    allowed = me.inputs(obs)["allowed"]
    choices = [None] + [(k, a) for k in range(len(CTRL)) for a in (HOLD, DUMP) if allowed[k, a]]

    def branch(c):
        if c is not None:
            acts = [FOLLOW] * len(CTRL)
            acts[c[0]] = c[1]
            me.override = acts
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
            env.step(*((a, b) if seat == 0 else (b, a)))
        return env.money[seat] - env.money[1 - seat]

    res = fork_branches(choices, branch, max_parallel=1)
    base = res[0]
    return {"job": j, "kind": kind, "opp": opp_name, "t": t_dec, "base": base,
            "dev": {f"{CTRL[c[0]]}:{'hold' if c[1] == HOLD else 'dump'}": r - base
                    for c, r in zip(choices[1:], res[1:]) if not isinstance(r, Exception)}}


def main():
    global VOCAB
    out, n = sys.argv[1], int(sys.argv[2])
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.environ.setdefault("ROUTE_MIX", "near=0.35,loss=0.15,top=0.15,live=0.35")
    if len(sys.argv) > 4:
        RD.POOLS.update(RD.load_mix(sys.argv[4]))
    VOCAB = Vocab(sys.argv[5] if len(sys.argv) > 5 else "data/opp_vocab.npz")
    start = int(os.environ.get("HEAD_START", "500000"))
    rows = []
    with warm_pool(procs, maxtasksperchild=4) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run_group, range(start, start + n)):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            best = max(r["dev"].items(), key=lambda kv: kv[1]) if r["dev"] else ("-", 0)
            print(json.dumps({"job": r["job"], "kind": r["kind"], "t": r["t"], "base": r["base"], "n_dev": len(r["dev"]),
                              "best": best}), flush=True)
    by = {}
    for r in rows:
        for k, v in r["dev"].items():
            by.setdefault(k, []).append(v)
    for k in sorted(by):
        v = np.array(by[k])
        print(json.dumps({"action": k, "n": len(v), "mean": round(float(v.mean())), "se": round(float(v.std() / np.sqrt(len(v)))),
                          "p_pos": round(float((v > 0).mean()), 3), "p_neg": round(float((v < 0).mean()), 3)}))
    bests = np.array([max([0] + list(r["dev"].values())) for r in rows])
    flips = np.mean([any((r["base"] + v > 0) != (r["base"] > 0) for v in r["dev"].values()) for r in rows])
    print(json.dumps({"groups": len(rows), "mean_best_gain": round(float(bests.mean())), "median_best_gain": float(np.median(bests)),
                      "frac_groups_with_win_flip": round(float(flips), 3)}))


if __name__ == "__main__":
    main()
