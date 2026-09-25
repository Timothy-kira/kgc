"""Closed-loop evaluation of a CLM checkpoint against opponent agents (both seats, several seeds).

python -m tools.eval_clm <weights.pt> <model_args.json> <opp1.py,opp2.py,...> [seeds=2] [procs=2] [temperature=0]
Opponents are Kaggle agent files (`agent(observation, configuration)`), loaded fresh for every game; the
special name `pass` is an always-PASS agent (sanity check). Uses the local FarmEnv (bit-exact with the
official interpreter). Prints one line per game and a summary per opponent: W/D/L, mean money, and our
per-step latency (2 torch threads).
"""
import importlib.util
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import uuid

import numpy as np

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_agent(path):
    if path == "pass":
        return lambda obs, cfg=None: PASS
    name = "opp_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    import inspect
    try:
        n = len(inspect.signature(m.agent).parameters)
    except (TypeError, ValueError):
        n = 2
    return m.agent if n >= 2 else (lambda obs, cfg=None, f=m.agent: f(obs))


def play_one(job):
    weights, args_json, opp, seed, seat, temp = job
    import torch
    torch.set_num_threads(2)
    from agent.clm_agent import CLMAgent, load_clm
    from env.fast_env import FarmEnv
    me = CLMAgent(load_clm(weights, args_json), temperature=temp, seed=seed)
    other = load_agent(opp)
    env = FarmEnv(seed)
    t0 = time.time()
    while not env.done:
        obs = [env.obs(0), env.obs(1)]
        acts = [None, None]
        acts[seat] = me(obs[seat])
        acts[1 - seat] = other(obs[1 - seat], env.config)
        env.step(acts[0], acts[1])
    m = env.money
    ts = np.array(me.times)
    return dict(opp=os.path.basename(opp), seed=seed, seat=seat, me=m[seat], them=m[1 - seat],
                mean_ms=round(ts.mean() * 1e3, 1), p99_ms=round(np.percentile(ts, 99) * 1e3, 1),
                max_ms=round(ts.max() * 1e3, 1), game_s=round(time.time() - t0))


def main():
    weights, args_json, opps = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    temp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    jobs = [(weights, args_json, o, 7000 + s, seat, temp) for o, s, seat in itertools.product(opps, range(seeds), (0, 1))]
    res = []
    with mp.get_context("spawn").Pool(procs) as p:
        for r in p.imap_unordered(play_one, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    print("=== summary (ours vs opponent) ===")
    for o in dict.fromkeys(r["opp"] for r in res):
        rs = [r for r in res if r["opp"] == o]
        w = sum(r["me"] > r["them"] for r in rs)
        d = sum(r["me"] == r["them"] for r in rs)
        print(json.dumps({"opp": o, "games": len(rs), "W": w, "D": d, "L": len(rs) - w - d,
                          "mean_me": round(np.mean([r["me"] for r in rs])),
                          "mean_them": round(np.mean([r["them"] for r in rs])),
                          "mean_ms": round(np.mean([r["mean_ms"] for r in rs]), 1),
                          "max_ms": max(r["max_ms"] for r in rs)}), flush=True)


if __name__ == "__main__":
    main()
