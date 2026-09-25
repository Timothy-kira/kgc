"""Relabeled (executed-only) market orders must reproduce the replay exactly: same money, hands and land at
every step for both players, and report how many orders were dropped.

python -m tools.test_relabel <replay_db_dir> [n_episodes=5] [min_score=2800]
"""
import json
import sys

from data.replay_db import ReplayDB, _unz
from data.seq_extract import effective_action
from env.fast_env import FarmEnv


def state(env):
    return [(round(env.money[p], 6), len(env.obs(p)["farms"][p]["hands"]),
             len(env.obs(p)["farms"][p].get("unlocked_quadrants") or [])) for p in (0, 1)]


def check(row):
    acts = _unz(row["actions_zstd"])
    cfg = {k: v for k, v in json.loads(row["config"]).items() if v is not None}
    tpd = int(cfg.get("turnsPerDay") or 24)
    ref, trace = FarmEnv(row["seed"], cfg), []
    trace.append(state(ref))
    for a0, a1 in acts:
        if ref.done:
            break
        ref.step(a0, a1)
        trace.append(state(ref))
    env, dropped, total = FarmEnv(row["seed"], cfg), 0, 0
    for t, pair in enumerate(acts):
        if env.done or t + 1 >= len(trace):
            break
        new = []
        for p in (0, 1):
            (_, hb, qb), (_, ha, qa) = trace[t][p], trace[t + 1][p]
            a = effective_action(pair[p], hb, ha, qb, qa, (t + 1) % tpd == 0)
            if isinstance(pair[p], dict):
                total += len(pair[p].get("market") or [])
                dropped += len(pair[p].get("market") or []) - len(a.get("market") or [])
            new.append(a)
        env.step(new[0], new[1])
        if state(env) != trace[t + 1]:
            return False, t, dropped, total
    return True, None, dropped, total


if __name__ == "__main__":
    db = ReplayDB(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else 2800
    eps = db.episodes(min_score=ms).sort_values("episode_id").tail(n)
    ok_all = True
    for eid in eps["episode_id"]:
        ok, t, d, tot = check(db.row(int(eid)))
        ok_all &= ok
        print(eid, "OK" if ok else f"MISMATCH at step {t}", f"dropped {d}/{tot} market orders", flush=True)
    print("RELABEL_EXACT", ok_all)
