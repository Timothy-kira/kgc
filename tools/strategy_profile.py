"""Reverse-engineer top teams' strategies from replays: per-team strategy profiles, compared with our agents.

python -m tools.strategy_profile <replay_db_dir> <out.json> [per_team=150] [procs=4] [compare=cha22,hakfield]
For each top team, its most recent episodes are re-simulated exactly (FarmEnv) and the team's seat is profiled:
  opening      orders of day 0 / day 1 (animals, seeds, hires, product buys, land)
  daily state  end-of-day money, hands, crop tiles per crop, animals per type, unlocked quadrants
  market       every SELL / BUY order: day, hour, item, quantity, visible price vs base price
  land         step of each BUY_LAND that executed
Local agents in `compare` are profiled on the same seeds (vs cha22) through the same code path.
"""
import collections
import json
import sys

import numpy as np
import pandas as pd

from agent.loader import load_agent
from data.replay_db import ReplayDB, _unz
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool

TOP_TEAMS = {16915014: "Boey", 16730612: "Unknown Mother-Goose", 16681125: "M & M & P & Q",
             16718819: "Majkel1337", 16732748: "DSM", 16770421: "Vadim Vasilenko", 16640510: "SpaTaro",
             16621799: "Artem The Farmer", 16730524: "THIRD FARM CLUB"}
BASE = {"WHEAT": 30, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 150, "MELON": 200, "EGG": 60, "MILK": 150,
        "WOOL": 200, "FERTILIZER": 10}
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
TPD = 24


def profile_game(seed, cfg, acts, seat):
    env = FarmEnv(seed, cfg)
    days, orders, land = [], [], []
    hands_max = collections.defaultdict(int)
    prev_q = 1
    for t, pair in enumerate(acts):
        if env.done:
            break
        o = env.obs(seat)
        f = o["farms"][seat]
        d = t // TPD
        hands_max[d] = max(hands_max[d], len(f["hands"]))
        a = pair[seat] if isinstance(pair[seat], dict) else {}
        for od in (a.get("market") or []):
            if isinstance(od, list) and od:
                item = od[1] if len(od) > 1 else None
                qty = od[2] if len(od) > 2 else 1
                orders.append({"t": t, "day": d, "hour": t % TPD, "op": od[0], "item": item,
                               "qty": qty if isinstance(qty, (int, float)) else 0,
                               "price": o["market"]["prices"].get(item) if item else None})
        if t % TPD == TPD - 1:
            cells = [c for row in f["tiles"] for c in row if isinstance(c, dict)]
            days.append({"day": d, "money": f["money"], "hands": hands_max[d],
                         "crops": {c: sum(1 for x in cells if x.get("crop") == c) for c in CROPS},
                         "animals": {a_: sum(1 for x in cells if x.get("animal") == a_) for a_ in ANIMALS},
                         "quads": len(f["unlocked_quadrants"])})
        env.step(pair[0], pair[1])
        q = len(env.obs(seat)["farms"][seat]["unlocked_quadrants"])
        if q > prev_q:
            land.append(t)
            prev_q = q
    m = env.money
    return {"days": days, "orders": orders, "land": land, "final": m[seat], "opp": m[1 - seat]}


def _replay_job(args):
    db_dir, eid, seat, team = args
    best_effort()
    db = ReplayDB(db_dir)
    row = db.row(eid)
    cfg = {k: v for k, v in json.loads(row["config"]).items() if v is not None}
    p = profile_game(row["seed"], cfg, _unz(row["actions_zstd"]), seat)
    p.update(team=team, eid=eid, seed=row["seed"], score=row[f"updated_score_{seat}"])
    return p


def _local_job(args):
    name, path, opp_path, seed, seat = args
    best_effort()
    me, opp = load_agent(path), load_agent(opp_path)
    env = FarmEnv(seed)
    acts = []
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a = [None, None]
        a[seat], a[1 - seat] = me(o[seat], env.config), opp(o[1 - seat], env.config)
        acts.append(a)
        env.step(a[0], a[1])
    p = profile_game(seed, {}, acts, seat)
    p.update(team=name, eid=-1, seed=seed, score=None)
    return p


def summarise(ps):
    """Aggregate profiles of one group into comparable statistics."""
    s = {"games": len(ps), "win_rate": float(np.mean([p["final"] > p["opp"] for p in ps])),
         "final_money": float(np.mean([p["final"] for p in ps]))}
    op = collections.Counter()
    for p in ps:
        for o in p["orders"]:
            if o["day"] == 0:
                key = o["op"] + (":" + o["item"] if o["item"] else "")
                op[key] += o["qty"] if o["op"] != "HIRE" else 1
    s["day0_orders_per_game"] = {k: round(v / len(ps), 1) for k, v in op.most_common(10)}
    s["land_day"] = [round(float(np.mean([p["land"][i] // TPD for p in ps if len(p["land"]) > i])), 1)
                     if any(len(p["land"]) > i for p in ps) else None for i in range(3)]
    s["land_frac"] = [round(float(np.mean([len(p["land"]) > i for p in ps])), 2) for i in range(3)]
    for ph, (a, b) in {"d0-9": (0, 10), "d10-19": (10, 20), "d20-29": (20, 30)}.items():
        ds = [d for p in ps for d in p["days"] if a <= d["day"] < b]
        if not ds:
            continue
        s[f"crops_{ph}"] = {c: round(float(np.mean([d["crops"][c] for d in ds])), 1) for c in CROPS}
        s[f"animals_{ph}"] = {x: round(float(np.mean([d["animals"][x] for d in ds])), 1) for x in ANIMALS}
        s[f"hands_{ph}"] = round(float(np.mean([d["hands"] for d in ds])), 1)
    money = np.array([[d["money"] for d in p["days"][:29]] for p in ps if len(p["days"]) >= 29])   # day 29 is partial
    if len(money):
        s["money_by_day"] = [round(float(x)) for x in money.mean(0)[[4, 9, 14, 19, 24, 28]]]
    sells = [o for p in ps for o in p["orders"] if o["op"] == "SELL" and o["qty"]]
    if sells:
        df = pd.DataFrame(sells)
        df["rel"] = df.apply(lambda r: (r["price"] or 0) / BASE.get(r["item"], 1), axis=1)
        vol = df.groupby("item")["qty"].sum()
        s["sell_volume_share"] = {k: round(float(v / vol.sum()), 3) for k, v in vol.sort_values(ascending=False).items()}
        s["sell_price_vs_base"] = {k: round(float(np.average(g["rel"], weights=g["qty"])), 2)
                                   for k, g in df.groupby("item")}
        s["sell_qty_median"] = {k: float(g["qty"].median()) for k, g in df.groupby("item")}
        h = df.groupby("hour")["qty"].sum()
        s["sell_hour_share"] = {int(k): round(float(v / h.sum()), 3) for k, v in h.items() if v / h.sum() > 0.03}
        dd = df.groupby(df["day"] // 5)["qty"].sum()
        s["sell_by_5day_block"] = [round(float(v / dd.sum()), 3) for v in dd.values]
    return s


def main():
    db_dir, out = sys.argv[1], sys.argv[2]
    per_team = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    compare = (sys.argv[5] if len(sys.argv) > 5 else "cha22,hakfield").split(",")
    e = ReplayDB(db_dir).episodes()
    jobs = []
    for tid, name in TOP_TEAMS.items():
        rows = []
        for k in (0, 1):
            sub = e[e[f"team_id_{k}"] == tid]
            rows += [(int(r.episode_id), k, float(getattr(r, f"updated_score_{k}") or 0)) for r in sub.itertuples()]
        rows = sorted(rows, reverse=True)[:per_team]                      # most recent episodes of the team
        jobs += [(db_dir, eid, k, name) for eid, k, _ in rows]
    seeds = sorted({int(x) for x in e.sort_values("episode_id").tail(40)["seed"]})[:20]
    path = lambda n: f"league/{n}.py" if n in ("cha22", "metav4") else f"league/pub/{n}/main.py"
    local = [(n, path(n), "league/cha22.py", s, seat) for n in compare for s in seeds for seat in (0, 1)]
    profiles = collections.defaultdict(list)
    with warm_pool(procs, maxtasksperchild=50) as pool:
        for p in pool.imap_unordered(_replay_job, jobs, chunksize=4):
            profiles[p["team"]].append(p)
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for p in pool.imap_unordered(_local_job, local):
            profiles["local:" + p["team"]].append(p)
    summary = {g: summarise(ps) for g, ps in profiles.items()}
    json.dump(summary, open(out, "w"), indent=1)
    for g, s in summary.items():
        print(g, json.dumps({k: s.get(k) for k in ("games", "win_rate", "final_money", "day0_orders_per_game",
                                                   "land_day", "money_by_day")}))


if __name__ == "__main__":
    main()
