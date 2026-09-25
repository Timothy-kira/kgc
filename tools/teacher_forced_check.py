"""Teacher-forced agreement of the *inference* path (CLMAgent: live Tracker features + incremental KV decode)
with a replay player's real decisions. If this matches the training accuracy, inference is consistent with
training and weak closed-loop play is a policy-quality problem, not a pipeline bug.

python -m tools.teacher_forced_check <weights.pt> <model_args.json> <replay_db_dir> [n_episodes=2] [min_score=2800]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from agent.action_space import SLOT_MARKET, action_to_decisions
from agent.clm_agent import CLMAgent, load_clm
from data.replay_db import ReplayDB


def run(weights, args_json, db_dir, n_ep=2, min_score=2800):
    torch.set_num_threads(2)
    ag = CLMAgent(load_clm(weights, args_json))
    net = ag.net
    db = ReplayDB(db_dir)
    eps = db.episodes(min_score=min_score)
    eps = eps.sort_values("episode_id").tail(n_ep)
    tot = {"unit": [0, 0], "market": [0, 0]}
    for eid in eps["episode_id"]:
        row = db.row(int(eid))
        acts = __import__("data.replay_db", fromlist=["_unz"])._unz(row["actions_zstd"])
        for seat in (0, 1):
            ag.reset()
            for t, env in ReplayDB.rebuild(None, int(eid), row=row):
                if t >= len(acts) or env.done:
                    break
                obs = env.obs(seat)
                a = acts[t][seat] if isinstance(acts[t][seat], dict) else {"farmer": ["PASS"], "hands": [], "market": []}
                n_hands = len(obs["farms"][int(obs["player"])]["hands"])
                with torch.inference_mode():
                    h = ag._feed(torch.cat([ag._obs_embeddings(obs), ag.act_emb], 0))
                    scale = float(net.scale())
                    for slot, c in action_to_decisions(a, n_hands):
                        zs = F.normalize(net.state_head(h).float(), dim=-1)
                        Z, enc, k = (ag.zm, ag.enc_m, "market") if slot == SLOT_MARKET else (ag.zu, ag.enc_u, "unit")
                        pred = int((scale * (Z @ zs)).argmax())
                        tot[k][0] += int(pred == c)
                        tot[k][1] += 1
                        h = ag._feed((ag.slot_emb[min(slot, 2)] + enc[c]).unsqueeze(0))
            print(eid, seat, {k: round(v[0] / max(v[1], 1), 4) for k, v in tot.items()}, flush=True)
    return {k: round(v[0] / max(v[1], 1), 4) for k, v in tot.items()}


if __name__ == "__main__":
    a = sys.argv[1:]
    print("TF_ACC", run(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 2, float(a[4]) if len(a) > 4 else 2800))
