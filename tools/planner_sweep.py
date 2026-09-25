"""Evaluate TopPlanner profile variants in parallel (warm fork pool) against an opponent.

python -m tools.planner_sweep <variants.json> [seeds=3] [procs=4] [opponent=league/cha22.py]
variants.json: {"name": {profile overrides}, ...}; "phases" may be given as a full list. Prints mean final money,
mean money difference and wins per variant (both seats).
"""
import json
import sys
import time

from agent.loader import load_agent
from agent.planner import TopPlanner
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool


def play(job):
    name, prof, seed, seat, opp_path = job
    best_effort()
    me, opp = TopPlanner(prof), load_agent(opp_path)
    env = FarmEnv(seed)
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a = me(o[seat], env.config)
        b = opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    m = env.money
    return name, m[seat], m[seat] - m[1 - seat]


def main():
    variants = json.load(open(sys.argv[1]))
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    opp = sys.argv[4] if len(sys.argv) > 4 else "league/cha22.py"
    for v in variants.values():
        if "phases" in v:
            v["phases"] = [(d, c, a) for d, c, a in v["phases"]]
    jobs = [(n, p, 9600 + s, seat, opp) for n, p in variants.items() for s in range(seeds) for seat in (0, 1)]
    t0 = time.time()
    res = {}
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for name, money, diff in pool.imap_unordered(play, jobs):
            res.setdefault(name, []).append((money, diff))
    for name in variants:
        r = res[name]
        print(json.dumps({"variant": name, "money": round(sum(m for m, _ in r) / len(r)),
                          "diff": round(sum(d for _, d in r) / len(r)), "wins": f"{sum(d > 0 for _, d in r)}/{len(r)}"}))
    print(f"{len(jobs)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
