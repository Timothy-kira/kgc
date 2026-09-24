"""Summarize strategic choices of players from replay DB rows or local games.

python -m tools.analyze_play <db_dir> [min_score]
"""
import collections
import json
import sys

from data.replay_db import ReplayDB

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def summarize_actions(acts, player, tpd=24):
    s = collections.Counter()
    sell_by_day = collections.defaultdict(lambda: collections.Counter())
    hires_by_day = collections.Counter()
    first = {}
    for t, pair in enumerate(acts):
        a = pair[player] or {}
        d = t // tpd
        for o in a.get("market", []) or []:
            if not isinstance(o, list) or not o:
                continue
            op = o[0]
            if op == "HIRE":
                hires_by_day[d] += 1
            elif op == "BUY_LAND":
                s["land"] += 1
                first.setdefault("land%d" % s["land"], d)
            elif op in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and len(o) >= 3:
                s[f"{op}:{o[1]}"] += int(o[2])
                first.setdefault(f"{op}:{o[1]}", d)
            elif op == "SELL" and len(o) >= 3:
                sell_by_day[d][o[1]] += 1  # number of sell orders issued (qty is often "sell all")
        for u in [a.get("farmer")] + list(a.get("hands", []) or []):
            if isinstance(u, list) and u and u[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE"):
                s[f"{u[0]}:{u[1] if len(u) > 1 else ''}"] += 1
    return s, sell_by_day, hires_by_day, first


def main():
    db = ReplayDB(sys.argv[1])
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    agg = {"win": collections.Counter(), "lose": collections.Counter()}
    n = {"win": 0, "lose": 0}
    hires = {"win": collections.Counter(), "lose": collections.Counter()}
    for r in db.iter_rows(min_score=ms):
        acts = r["actions"]
        for p in (0, 1):
            me, op = r[f"reward_{p}"], r[f"reward_{1 - p}"]
            if me is None or op is None or me == op:
                continue
            k = "win" if me > op else "lose"
            s, sbd, hbd, first = summarize_actions(acts, p)
            agg[k].update(s)
            agg[k]["money"] += me
            n[k] += 1
            for d, h in hbd.items():
                hires[k][d // 5] += h
    for k in agg:
        print(f"== {k} (n={n[k]})  avg money {agg[k]['money'] / max(1, n[k]):.0f}")
        for key, v in sorted(agg[k].items(), key=lambda kv: -kv[1]):
            if key != "money":
                print(f"   {key:28s} {v / max(1, n[k]):8.1f}")
        print("   hires per 5-day block:", {b: round(v / max(1, n[k]) / 5, 1) for b, v in sorted(hires[k].items())})


if __name__ == "__main__":
    main()
