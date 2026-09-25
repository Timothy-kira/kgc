"""Knob insertion point: which of cha22's disabled chassis layers help against the ladder distribution?

python -m tools.settings_sweep <out.jsonl> [n_games=120] [procs=4] [db_dir]
cha22 ships `_SETTINGS` with budget_guard, room_guard, clamp_sells, dead_stock, terminal_liquidation (and
front_run, which needs an opponent plan) switched off. Each variant flips one layer on; every variant plays the
SAME games (paired design): opponents drawn from the RL mix (near-rating ladder tapes, top tapes, live league),
both seats. Reports win rate, mean money difference and the paired difference vs default cha22.
"""
import json
import os
import sys

import numpy as np

import tools.route_data as RD
from agent.loader import call_adapter, entry_name, load_agent, load_module
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool

VARIANTS = {"default": {}, "budget_guard": {"budget_guard": True}, "room_guard": {"room_guard": True},
            "clamp_sells": {"clamp_sells": True}, "dead_stock": {"dead_stock": True},
            "terminal_liquidation": {"terminal_liquidation": True}}


def play(job):
    name, j = job
    best_effort()
    kind, opp_name, seed, seat, cfg, tape = RD.job_spec(j)
    m = load_module(RD.CHA22)
    m._IMPL.chassis.cfg.update(VARIANTS[name])
    me = call_adapter(getattr(m, entry_name(m)))
    opp = tape if tape is not None else load_agent(opp_name)
    env = FarmEnv(seed, cfg)
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    return name, j, kind, env.money[seat] - env.money[1 - seat]


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if len(sys.argv) > 4:
        RD.POOLS.update(RD.load_mix(sys.argv[4]))
    os.environ.setdefault("ROUTE_MIX", "near=0.5,top=0.2,live=0.3")
    jobs = [(v, 300000 + j) for j in range(n) for v in VARIANTS]
    res = {}
    with warm_pool(procs, maxtasksperchild=4) as pool, open(out, "a") as f:
        for name, j, kind, d in pool.imap_unordered(play, jobs):
            res.setdefault(name, {})[j] = (kind, d)
            f.write(json.dumps({"variant": name, "job": j, "kind": kind, "diff": d}) + "\n")
    base = res["default"]
    for name, r in res.items():
        js = [j for j in r if j in base]
        d = np.array([r[j][1] for j in js])
        pd = np.array([r[j][1] - base[j][1] for j in js])
        by = {}
        for k in ("near", "top", "live"):
            sel = [r[j][1] > 0 for j in js if r[j][0] == k]
            if sel:
                by[k] = round(float(np.mean(sel)), 3)
        print(json.dumps({"variant": name, "n": len(js), "win": round(float(np.mean(d > 0)), 3),
                          "mean_diff": round(float(d.mean())), "paired_vs_default": round(float(pd.mean())),
                          "paired_se": round(float(pd.std() / np.sqrt(max(1, len(pd))))), "win_by_kind": by}), flush=True)


if __name__ == "__main__":
    main()
