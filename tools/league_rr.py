"""Round robin between agent files with Kaggle's real entry points, on a warm fork pool (infra/fork.py).

python -m tools.league_rr <a.py,b.py,...> [seeds=8] [procs=4]
Every unordered pair plays both seats on each seed. Workers are forked after the agent sources are compiled
once in the parent (DSec-style warm pool); bulk games run under SCHED_IDLE. Prints per-agent W/D/L, mean
money diff and a pairwise table.
"""
import itertools
import json
import sys
import time

from agent.loader import load_agent
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool


def play(job):
    a, b, seed = job
    best_effort()
    fa, fb = load_agent(a), load_agent(b)
    env = FarmEnv(seed)
    while not env.done:
        env.step(fa(env.obs(0), env.config), fb(env.obs(1), env.config))
    return a, b, seed, env.money[0], env.money[1]


def main():
    paths = sys.argv[1].split(",")
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    for p in paths:                         # compile once in the parent -> .pyc shared by forked workers
        load_agent(p)
    jobs = [(a, b, 9000 + s) for a, b in itertools.permutations(paths, 2) for s in range(seeds)]
    t0 = time.time()
    stat = {p: [0, 0, 0, 0.0] for p in paths}
    pair = {}
    with warm_pool(procs, maxtasksperchild=1) as pool:   # fresh fork per game (multi-file agents)
        for a, b, seed, ma, mb in pool.imap_unordered(play, jobs):
            for x, y, mx, my in ((a, b, ma, mb), (b, a, mb, ma)):
                s = stat[x]
                s[0 if mx > my else 1 if mx == my else 2] += 1
                s[3] += mx - my
                pair.setdefault((x, y), []).append(mx - my)
    name = lambda p: (p.rsplit("/", 2)[-2] if p.endswith("/main.py") else p.rsplit("/", 1)[-1].replace(".py", ""))[:40]
    for p in sorted(paths, key=lambda p: -stat[p][3]):
        w, d, l, tot = stat[p]
        print(json.dumps({"agent": name(p), "W": w, "D": d, "L": l, "mean_diff": round(tot / max(w + d + l, 1))}))
    for a, b in itertools.combinations(paths, 2):
        v = pair[(a, b)]
        print(f"{name(a):>10} vs {name(b):<10} wins {sum(x > 0 for x in v)}/{len(v)} mean diff {sum(v) / len(v):+.0f}")
    print(f"{len(jobs)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
