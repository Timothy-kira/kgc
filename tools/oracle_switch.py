"""Upper-bound study for expert routing: greedy per-day switching between whole-agent experts vs an opponent.

python -m tools.oracle_switch <opponent.py> <seed> [experts=metav4,farm2945,cha22,v48,salem2900] [procs=4] [base=metav4]
A driver keeps ONE live game: the env, every expert (run in shadow mode every step so their trackers stay in
sync) and the opponent. At each day boundary it forks one child per expert (infra/fork.py: copy-on-write
snapshot of the whole Python state, DSec-style checkpoint/branch); child k plays that day with expert k and
the base expert afterwards and reports the final money difference. The driver then commits the best expert for
the day and advances. No prefix replay, no process spawn or re-import per candidate.
"""
import json
import os
import sys
import time

from agent.loader import load_agent
from agent.skills import LayeredExpert
from env.fast_env import FarmEnv
from infra.fork import fork_branches

TPD = 24


class Game:
    def __init__(self, opp_path, seed, expert_paths):
        self.experts = [LayeredExpert(p, record_layers=False) for p in expert_paths]
        self.opp = load_agent(opp_path)
        self.env = FarmEnv(seed)

    def step(self, k):
        env = self.env
        o0, o1 = env.obs(0), env.obs(1)
        acts = [e(o0, env.config)[0] for e in self.experts]          # shadow: every expert sees every step
        env.step(acts[k], self.opp(o1, env.config))

    def play_until(self, schedule_fn, stop_step=None):
        while not self.env.done and (stop_step is None or self.env.step_count < stop_step):
            self.step(schedule_fn(self.env.step_count // TPD))

    def diff(self):
        return self.env.money[0] - self.env.money[1], self.env.money[0], self.env.money[1]


def main():
    opp, seed = sys.argv[1], int(sys.argv[2])
    names = (sys.argv[3] if len(sys.argv) > 3 else "metav4,farm2945,cha22,v48,salem2900").split(",")
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    base = names.index(sys.argv[5] if len(sys.argv) > 5 else "metav4")
    t0 = time.time()
    path = lambda n: n if n.endswith(".py") else (f"league/{n}.py" if os.path.exists(f"league/{n}.py") else f"league/pub/{n}/main.py")
    g = Game(opp, seed, [path(n) for n in names])

    def static(k):
        g.play_until(lambda d: k)
        return g.diff()
    for n, r in zip(names, fork_branches(list(range(len(names))), static, procs)):
        print(json.dumps({"static": n, "diff": round(r[0]), "me": round(r[1]), "opp": round(r[2])}), flush=True)

    sched = {}
    for d in range(30):
        if g.env.done:
            break

        def branch(k, d=d):
            g.play_until(lambda dd: k if dd == d else base)
            return g.diff()
        res = fork_branches(list(range(len(names))), branch, procs)
        k = max(range(len(names)), key=lambda i: res[i][0])
        sched[d] = k
        g.play_until(lambda dd: k, stop_step=(d + 1) * TPD)           # commit the day in the driver
        print(json.dumps({"day": d, "pick": names[k], "diff_if_base_after": round(res[k][0]),
                          "alts": [round(r[0]) for r in res], "t": round(time.time() - t0)}), flush=True)
    g.play_until(lambda d: base)
    fin = g.diff()
    print(json.dumps({"greedy_diff": round(fin[0]), "me": round(fin[1]), "opp": round(fin[2]),
                      "schedule": [names[sched.get(d, base)] for d in range(30)],
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
