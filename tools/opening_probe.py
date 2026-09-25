"""Side-by-side opening of a CLM checkpoint vs a replay player (same seed, same opponent actions open-loop).

python -m tools.opening_probe <weights.pt> <model_args.json> <replay_db_dir> [episode_id] [steps=30] [seat=0]
Prints per step: money, #hands, farmer action and market orders, first for the replay player, then for the
model. Also summarises market-op counts over the probed steps. Shows *where* the policy diverges (e.g. never
buying seeds/animals in the first steps), which the final-money check cannot.
"""
import collections
import json
import sys

import torch

from agent.clm_agent import CLMAgent, load_clm
from data.replay_db import ReplayDB, _unz
from env.fast_env import FarmEnv


def trace(env, policy, opp_acts, seat, steps):
    rows, ops = [], collections.Counter()
    for t in range(steps):
        if env.done:
            break
        o = env.obs(seat)
        a = policy(t, o)
        rows.append((t, round(env.money[seat]), len(o["farms"][seat]["hands"]), a.get("farmer"), a.get("market")))
        for m in a.get("market") or []:
            ops[m[0]] += 1
        pair = [None, None]
        pair[seat], pair[1 - seat] = a, opp_acts[t][1 - seat]
        env.step(pair[0], pair[1])
    return rows, ops, round(env.money[seat])


def main():
    w, aj, db_dir = sys.argv[1:4]
    db = ReplayDB(db_dir)
    eid = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else \
        int(db.episodes(min_score=2800).sort_values("episode_id")["episode_id"].iloc[-1])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    seat = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    torch.set_num_threads(2)
    row = db.row(eid)
    acts = _unz(row["actions_zstd"])
    cfg = {k: v for k, v in json.loads(row["config"]).items() if v is not None}
    pas = {"farmer": ["PASS"], "hands": [], "market": []}
    replay = lambda t, o: acts[t][seat] if isinstance(acts[t][seat], dict) else pas
    me = CLMAgent(load_clm(w, aj))
    model = lambda t, o: me(o)
    for name, pol in (("replay", replay), ("model", model)):
        rows, ops, final = trace(FarmEnv(row["seed"], cfg), pol, acts, seat, steps)
        print(f"--- {name} (episode {eid}, seat {seat})")
        for r in rows:
            print(r[0], r[1], r[2], json.dumps(r[3]), json.dumps(r[4])[:150])
        print(f"{name}: money after {steps} steps = {final}; market ops {dict(ops)}", flush=True)


if __name__ == "__main__":
    main()
