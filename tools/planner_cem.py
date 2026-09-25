"""Cross-entropy search over TopPlanner parameters (heuristics tuned by optimisation), warm fork pool.

python -m tools.planner_cem <out.json> [iters=20] [pop=24] [seeds=3] [procs=4] [opponent=league/cha22.py]
Each candidate plays both seats on `seeds` seeds against the opponent; fitness = mean final money (primary)
+ 0.2 * mean money difference. The elite (top 25%) refits a diagonal Gaussian over normalised parameters.
Writes the best profile found so far to out.json after every iteration (resumable by re-reading it).
"""
import json
import random
import sys
import time

import numpy as np

from agent.loader import load_agent
from agent.planner import PROFILE, TopPlanner
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool

# name: (low, high, integer?)
SPACE = {
    "melon0": (6, 16, True), "wheat0": (4, 14, True), "straw3": (0, 12, True),
    "wheat_mid": (10, 35, True), "straw_mid": (5, 30, True), "tomato_mid": (0, 16, True), "carrot_mid": (0, 12, True),
    "wheat_late": (10, 35, True), "carrot_late": (0, 20, True), "tomato_late": (0, 14, True), "straw_late": (0, 20, True),
    "cow": (2, 10, True), "sheep": (0, 8, True), "goose": (0, 10, True),
    "land2": (5, 14, True), "land3": (8, 30, True),
    "zone_penalty": (0, 8, False), "hire_div": (12, 30, False), "hire_cap": (6, 13, True),
    "water_daily": (0, 10, False), "fert_min_price": (20, 70, False), "reserve": (50, 400, False),
}


def to_profile(v):
    g = lambda k: int(round(v[k])) if SPACE[k][2] else float(v[k])
    early_a = {"COW": min(2, g("cow")), "SHEEP": min(2, g("sheep"))}
    mid_a = {"COW": g("cow"), "SHEEP": g("sheep"), "GOOSE": g("goose")}
    return {
        "phases": [
            (0, {"MELON": g("melon0"), "WHEAT": g("wheat0")}, early_a),
            (3, {"MELON": g("melon0"), "WHEAT": g("wheat0") + 2, "STRAWBERRY": g("straw3")},
             {k: max(early_a.get(k, 0), (v + 1) // 2) for k, v in mid_a.items()}),
            (8, {"WHEAT": g("wheat_mid"), "STRAWBERRY": g("straw_mid"), "TOMATO": g("tomato_mid"),
                 "CARROT": g("carrot_mid")}, mid_a),
            (20, {"WHEAT": g("wheat_late"), "CARROT": g("carrot_late"), "TOMATO": g("tomato_late"),
                  "STRAWBERRY": g("straw_late")}, mid_a),
        ],
        "land_days": (6, g("land2"), g("land3")),
        "zone_penalty": g("zone_penalty"), "hire_div": g("hire_div"), "hire_cap": g("hire_cap"),
        "water_daily": g("water_daily"), "fert_min_price": g("fert_min_price"), "reserve": g("reserve"),
    }


def from_default():
    p = PROFILE
    ph = {d: (c, a) for d, c, a in p["phases"]}
    return {"melon0": 12, "wheat0": 8, "straw3": 6, "wheat_mid": 22, "straw_mid": 20, "tomato_mid": 10, "carrot_mid": 5,
            "wheat_late": 25, "carrot_late": 14, "tomato_late": 8, "straw_late": 10, "cow": 6, "sheep": 4, "goose": 6,
            "land2": p["land_days"][1], "land3": min(30, p["land_days"][2]), "zone_penalty": p["zone_penalty"],
            "hire_div": p["hire_div"], "hire_cap": p["hire_cap"], "water_daily": p["water_daily"],
            "fert_min_price": p["fert_min_price"], "reserve": p["reserve"]}


def play(job):
    ci, prof, seed, seat, opp_path = job
    best_effort()
    me, opp = TopPlanner(prof), load_agent(opp_path)
    env = FarmEnv(seed)
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a = me(o[seat], env.config)
        b = opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    m = env.money
    return ci, m[seat], m[seat] - m[1 - seat]


def main():
    out = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    opp = sys.argv[6] if len(sys.argv) > 6 else "league/cha22.py"
    keys = list(SPACE)
    lo = np.array([SPACE[k][0] for k in keys], float)
    hi = np.array([SPACE[k][1] for k in keys], float)
    d0 = from_default()
    mu = (np.array([d0[k] for k in keys], float) - lo) / (hi - lo)
    sd = np.full(len(keys), 0.2)
    best = (-1e18, None)
    rng = np.random.default_rng(0)
    for it in range(iters):
        t0 = time.time()
        cands = [mu.copy()] + [np.clip(mu + sd * rng.standard_normal(len(keys)), 0, 1) for _ in range(pop - 1)]
        vals = [{k: lo[i] + c[i] * (hi[i] - lo[i]) for i, k in enumerate(keys)} for c in cands]
        profs = [to_profile(v) for v in vals]
        s0 = 9700 + it * 17                                     # fresh seeds every iteration (no overfitting)
        jobs = [(ci, p, s0 + s, seat, opp) for ci, p in enumerate(profs) for s in range(seeds) for seat in (0, 1)]
        fit = {i: [] for i in range(pop)}
        with warm_pool(procs, maxtasksperchild=4) as pool:
            for ci, money, diff in pool.imap_unordered(play, jobs):
                fit[ci].append(money + 0.2 * diff)
        f = np.array([np.mean(fit[i]) for i in range(pop)])
        elite = np.argsort(-f)[: max(2, pop // 4)]
        mu = np.mean([cands[i] for i in elite], axis=0)
        sd = np.maximum(0.03, np.std([cands[i] for i in elite], axis=0))
        if f[elite[0]] > best[0]:
            best = (float(f[elite[0]]), vals[elite[0]])
        json.dump({"iter": it, "best_fitness": best[0], "best_params": best[1], "mean_params":
                   {k: lo[i] + mu[i] * (hi[i] - lo[i]) for i, k in enumerate(keys)},
                   "incumbent_fitness": float(f[0])}, open(out, "w"), indent=1)
        print(json.dumps({"iter": it, "incumbent": round(float(f[0])), "best_this_iter": round(float(f[elite[0]])),
                          "best_ever": round(best[0]), "sec": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
