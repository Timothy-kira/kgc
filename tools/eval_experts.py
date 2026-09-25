"""Value of each routed expert when forced on (gate always picks it where it proposes), on top of the shared
expert, against a panel of strong opponents (both seats, several seeds). Warm fork pool, SCHED_IDLE.

python -m tools.eval_experts [seeds=4] [procs=4] [opponents=cha22,prvsiyan_frontier,hakfield,tetsutani_ms,pilkwang_sep]
"""
import json
import sys
import time

from agent.loader import load_agent
from agent.top_experts import apply, default_experts, demand_experts
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool


EXPERTS = demand_experts if "--demand" in sys.argv else default_experts


def path(n):
    return f"league/{n}.py" if n in ("cha22", "metav4") else f"league/pub/{n}/main.py"


def play(job):
    k, opp, seed, seat = job
    best_effort()
    shared, other = load_agent("league/cha22.py"), load_agent(path(opp))
    experts = EXPERTS()
    env = FarmEnv(seed)
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        s = shared(o[seat], env.config)
        a = apply(experts[k](o[seat], s), s) if k >= 0 else s
        b = other(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    m = env.money
    return k, opp, m[seat] - m[1 - seat]


def main():
    argv = [a for a in sys.argv if a != "--demand"]
    seeds = int(argv[1]) if len(argv) > 1 else 4
    procs = int(argv[2]) if len(argv) > 2 else 4
    opps = (argv[3] if len(argv) > 3 else "cha22,prvsiyan_frontier,hakfield,tetsutani_ms,pilkwang_sep").split(",")
    names = ["cha22 (shared only)"] + [e.name for e in EXPERTS()]
    jobs = [(k, o, 9300 + s, seat) for k in range(-1, len(names) - 1) for o in opps for s in range(seeds) for seat in (0, 1)]
    t0 = time.time()
    res = {}
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for k, opp, d in pool.imap_unordered(play, jobs):
            res.setdefault(k, {}).setdefault(opp, []).append(d)
    base = {o: sum(v) / len(v) for o, v in res[-1].items()}
    for k in range(-1, len(names) - 1):
        r = res[k]
        allv = [x for v in r.values() for x in v]
        row = {"expert": names[k + 1], "wins": f"{sum(x > 0 for x in allv)}/{len(allv)}",
               "mean_diff": round(sum(allv) / len(allv)),
               "vs_base": round(sum(allv) / len(allv) - sum(base.values()) / len(base)),
               "by_opp": {o: round(sum(v) / len(v)) for o, v in r.items()}}
        print(json.dumps(row))
    print(f"{len(jobs)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
