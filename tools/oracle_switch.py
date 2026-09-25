"""Upper-bound study for expert routing: greedy per-day switching between whole-agent experts vs an opponent.

python -m tools.oracle_switch <opponent.py> <seed> [experts=metav4,farm2945,cha22,v48,salem2900] [procs=4] [base=metav4]
For day d = 0..29: for each expert k, replay the game from the start with the already fixed choices for days
< d, expert k for day d and the base expert afterwards; keep the k with the best final money difference.
All experts run in shadow mode every step (their trackers stay in sync), so switching at a day boundary is
valid. Replaying from the start avoids having to clone expert internals. Prints static baselines and the
greedy result (money diff vs opponent).
"""
import json
import multiprocessing as mp
import sys

from agent.skills import LayeredExpert, _load
from env.fast_env import FarmEnv

TPD = 24


def play(args):
    opp_path, seed, expert_paths, schedule, base = args      # schedule: {day: expert index}
    experts = [LayeredExpert(p, record_layers=False) for p in expert_paths]
    opp = _load(opp_path).agent
    env = FarmEnv(seed)
    while not env.done:
        o0, o1 = env.obs(0), env.obs(1)
        acts = [e(o0, env.config)[0] for e in experts]          # shadow: every expert sees every step
        k = schedule.get(env.step_count // TPD, base)
        env.step(acts[k], opp(o1, env.config))
    return env.money[0] - env.money[1], env.money[0], env.money[1]


def main():
    opp, seed = sys.argv[1], int(sys.argv[2])
    names = (sys.argv[3] if len(sys.argv) > 3 else "metav4,farm2945,cha22,v48,salem2900").split(",")
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    base_name = sys.argv[5] if len(sys.argv) > 5 else "metav4"
    paths = [f"league/{n}.py" for n in names]
    base = names.index(base_name)
    with mp.get_context("spawn").Pool(procs) as pool:
        static = pool.map(play, [(opp, seed, paths, {d: k for d in range(30)}, base) for k in range(len(paths))])
        for n, r in zip(names, static):
            print(json.dumps({"static": n, "diff": round(r[0]), "me": round(r[1]), "opp": round(r[2])}), flush=True)
        sched, best = {}, None
        for d in range(30):
            res = pool.map(play, [(opp, seed, paths, {**sched, d: k}, base) for k in range(len(paths))])
            k = max(range(len(paths)), key=lambda i: res[i][0])
            sched[d], best = k, res[k]
            print(json.dumps({"day": d, "pick": names[k], "diff": round(best[0]),
                              "alts": [round(r[0]) for r in res]}), flush=True)
    print(json.dumps({"greedy_diff": round(best[0]), "me": round(best[1]), "opp": round(best[2]),
                      "schedule": [names[sched[d]] for d in range(30)]}), flush=True)


if __name__ == "__main__":
    main()
