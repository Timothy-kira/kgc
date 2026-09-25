"""Where does an agent lose value? Event accounting over a full game (by diffing consecutive states).

python -m tools.planner_diag [agent=planner] [opponent=league/cha22.py] [seeds=9301,9302]
Counts per game: plantings by crop, harvested units by product, plants lost to weeds (missed watering),
one-time crops that decayed (missed harvest window), animals escaped (unfed), items discarded by a full shed
(end-of-day overflow), revenue by product, spending (seeds / animals / hires / land / wheat), idle unit-steps,
and the money curve. Runs the same accounting for cha22 as reference.
"""
import collections
import json
import sys

from agent.loader import load_agent
from agent.planner import TopPlanner
from env.fast_env import FarmEnv

TPD = 24


def snapshot(o):
    f = o["farms"][int(o["player"])]
    tiles = {(x, y): (dict(c) if isinstance(c, dict) else c) for y, row in enumerate(f["tiles"]) for x, c in enumerate(row)}
    priv = o["private"]
    stored = sum(priv["shed"].values()) + sum(sum(i.values()) for i in priv["inventories"])
    return tiles, dict(priv["shed"]), [dict(i) for i in priv["inventories"]], f["money"], stored


def run(agent, opp, seed):
    env = FarmEnv(seed)
    ev = collections.Counter()
    money_curve = []
    prev = None
    while not env.done:
        o0 = env.obs(0)
        a = agent(o0, env.config)
        b = opp(env.obs(1), env.config)
        units = [a.get("farmer")] + list(a.get("hands") or [])
        for u in units:
            op = u[0] if isinstance(u, list) and u else "PASS"
            ev["act:" + op] += 1
            if op == "PLANT":
                ev["plant:" + u[1]] += 1
        for m in a.get("market") or []:
            if m and m[0] in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL") and len(m) >= 3:
                ev[f"order:{m[0]}:{m[1]}"] += int(m[2])
            elif m:
                ev["order:" + m[0]] += 1
        before = snapshot(o0)
        t = env.step_count
        env.step(a, b)
        after = snapshot(env.obs(0))
        tb, ta = before[0], after[0]
        for p, c in tb.items():
            n = ta.get(p)
            if isinstance(c, dict) and c.get("kind") == "PLANT":
                if isinstance(n, dict) and n.get("kind") == "WEED":
                    if c.get("consecutive_unwatered", 0) >= 1 and not c.get("watered_today"):
                        ev["lost:unwatered_to_weed:" + c["crop"]] += 1
                    else:
                        ev["end_of_life_to_weed:" + c["crop"]] += 1
                elif (isinstance(n, dict) and n.get("kind") == "PLANT" and n.get("yield_units", 0) < c.get("yield_units", 0)
                      and a is not None and not any(isinstance(u, list) and u and u[0] == "HARVEST" for u in units)):
                    ev["lost:decay_units:" + c["crop"]] += c["yield_units"] - n["yield_units"]
            if isinstance(c, dict) and "animal" in c and isinstance(n, dict) and "animal" not in n and n.get("kind") in ("COOP", "PASTURE"):
                ev["lost:animal_escaped:" + c["animal"]] += 1
        # end-of-day overflow: stored items vanish at the day boundary beyond what was sold/consumed
        if (t + 1) % TPD == 0:
            sold = sum(int(m[2]) for m in (a.get("market") or []) if m and m[0] == "SELL" and len(m) >= 3)
            drop = before[4] - after[4]
            if drop > sold + 8:
                ev["lost:overflow_items(approx)"] += drop - sold
            live = sum(1 for c in ta.values() if isinstance(c, dict) and c.get("kind") == "PLANT")
            money_curve.append((round(after[3]), live, sum(after[1].values())))
        d_money = after[3] - before[3]
        if d_money > 0:
            ev["revenue"] += d_money
    ev["final_money"] = round(env.money[0])
    ev["opp_money"] = round(env.money[1])
    return ev, money_curve


def main():
    who = sys.argv[1] if len(sys.argv) > 1 else "planner"
    opp_path = sys.argv[2] if len(sys.argv) > 2 else "league/cha22.py"
    seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "9301,9302").split(",")]
    for name in (who, "cha22"):
        tot = collections.Counter()
        curves = []
        for s in seeds:
            ag = TopPlanner() if name == "planner" else load_agent(f"league/{name}.py" if name == "cha22" else name)
            ev, curve = run(ag, load_agent(opp_path), s)
            tot.update(ev)
            curves.append(curve)
        n = len(seeds)
        keys = sorted(k for k in tot if not k.startswith("act:"))
        print(f"===== {name} (mean over {n} games)")
        print(json.dumps({k: round(tot[k] / n) for k in keys}))
        print("actions:", json.dumps({k[4:]: round(tot[k] / n) for k in sorted(tot, key=lambda k: -tot[k]) if k.startswith("act:")}))
        print("money by day:", curves[0][::3])


if __name__ == "__main__":
    main()
