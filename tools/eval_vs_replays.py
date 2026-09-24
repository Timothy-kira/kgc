"""Evaluate an agent against open-loop top-player replays.

python -m tools.eval_vs_replays <db_dir> <agent.py|ctrl:weights.npz> [n_episodes] [min_score]
For each episode, our agent replaces each seat in turn. Reports win rate and money gap,
and compares with how the replaced player actually did online.
"""
import multiprocessing as mp
import sys

import numpy as np

from data.replay_db import ReplayDB
from rl.replay_opponent import play_vs_replay


def make_agent(spec):
    from tools.arena import load
    if spec.startswith("ctrl:"):
        from agent.controller import MarketController
        from model.policy_np import NumpyPolicy
        pol = NumpyPolicy(spec[5:])
        return MarketController(load("/home/user/kgc/league/metav4.py"), pol.act_greedy)
    return load(spec)


def job(args):
    spec, row, seat = args
    ours, rep, rep_online, replaced_online = play_vs_replay(make_agent(spec), row, seat)
    return row["episode_id"], seat, ours, rep, rep_online, replaced_online


def main():
    db = ReplayDB(sys.argv[1])
    spec = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ms = float(sys.argv[4]) if len(sys.argv) > 4 else 3000
    rows = []
    for r in db.iter_rows(min_score=ms):
        if r["first_bad_step"] == -1:
            rows.append(r)
        if len(rows) >= n:
            break
    jobs = [(spec, r, s) for r in rows for s in (0, 1)]
    with mp.Pool(4) as p:
        res = p.map(job, jobs)
    ours = np.array([x[2] for x in res]); rep = np.array([x[3] for x in res])
    repl = np.array([x[5] for x in res]); rep_on = np.array([x[4] for x in res])
    w = (ours > rep).mean() + 0.5 * (ours == rep).mean()
    w_online = (repl > rep_on).mean() + 0.5 * (repl == rep_on).mean()
    print(f"{spec}: games={len(res)} winrate_vs_top={w:.3f} (replaced player online: {w_online:.3f}) "
          f"mean_gap={np.mean(ours - rep):.0f} our_money={ours.mean():.0f} replay_money={rep.mean():.0f}")


if __name__ == "__main__":
    main()
