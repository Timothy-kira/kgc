"""Check the CLM policy against the competition time limits on the Kaggle agent runtime profile.

Kaggle agent runtime (probe submission): torch 2.6 CPU, 2 threads. Limits: actTimeout 1 s per step
(+ 60 s overage for the whole game), runTimeout 1200 s per game.

python -m tools.bench_latency <weights.pt> <model_args.json> [steps=719] [threads=2]
Plays a full game (opponent passes unless one is given) and prints one JSON line: mean / p50 / p99 / max step latency, projected
game time, number of steps over 1 s and the overage consumed, and a PASS/FAIL verdict.
"""
import json
import sys
import time

import numpy as np
import torch

from agent.clm_agent import CLMAgent, load_clm
from env.fast_env import FarmEnv

ACT_TIMEOUT, OVERAGE, RUN_TIMEOUT = 1.0, 60.0, 1200.0


def bench(weights, args_json, steps=719, threads=2, seed=123, opponent=None):
    torch.set_num_threads(threads)
    net = load_clm(weights, args_json)
    ag = CLMAgent(net, temperature=0.0)
    env = FarmEnv(seed)
    t0 = time.time()
    for t in range(steps):
        if env.done:
            break
        a = ag(env.obs(0))
        b = opponent(env.obs(1)) if opponent else {"farmer": ["PASS"], "hands": [], "market": []}
        env.step(a, b)
    total = time.time() - t0
    ts = np.array(ag.times)
    over = np.clip(ts - ACT_TIMEOUT, 0, None)
    res = {"latency": True, "steps": len(ts), "mean_ms": round(ts.mean() * 1e3, 1),
           "p50_ms": round(np.percentile(ts, 50) * 1e3, 1), "p99_ms": round(np.percentile(ts, 99) * 1e3, 1),
           "max_ms": round(ts.max() * 1e3, 1), "game_s": round(total, 1),
           "projected_719_s": round(ts.mean() * 719, 1), "steps_over_1s": int((ts > ACT_TIMEOUT).sum()),
           "overage_used_s": round(float(over.sum()), 2), "threads": threads,
           "money": env.money}
    res["verdict"] = "PASS" if (res["overage_used_s"] < OVERAGE * 0.5 and res["projected_719_s"] < RUN_TIMEOUT * 0.6
                                and res["p99_ms"] < 800) else "FAIL"
    return res


if __name__ == "__main__":
    w, a = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 719
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    print(json.dumps(bench(w, a, steps, threads)), flush=True)
