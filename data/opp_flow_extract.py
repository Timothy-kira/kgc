"""Replay DB shard -> opponent event streams + targets for the Engram opponent memory (docs/PLAN_v4.1.md).

python -m data.opp_flow_extract <db_dir> <out_dir> [procs=4] [max_shards]
Environment-only re-simulation with the official interpreter (no observation tokens). For each episode and each
seat k, the opponent is seat 1-k. One npz per DB shard (finished shards are skipped), K trajectories:
    episode_id i64 [K]  seat i8 [K]  t_off i32 [K+1]  d_off i32 [K+1]
    s_code i64 [sumT]          opponent's per-step S code (agent/opp_events.py)
    flow i16 [sumT, 9]         opponent's exact inventory-moving flow per step (engine tap; live play infers it)
    opp_shed i16 [sumT, 9]     opponent's shed after the step (hidden-stock target)
    d_code i64 [sumD]          fingerprint code (step 2) + one D code per finished day
Trajectories line up with data/seq_extract.py shards through (episode_id, seat).
"""
import collections
import importlib
import multiprocessing as mp
import os
import sys

import numpy as np

from agent.opp_events import PRODUCTS, OppEventStream
from data.replay_db import ReplayDB

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
_orig_commit = K._commit_unit
TAP = {"env": None, "flows": None}


def _tap(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig_commit(op, item, price, farm, private, market, shed_capacity)
    fl = TAP["flows"]
    if ok and fl is not None and op in ("SELL", "BUY_PRODUCT"):
        pl = 0 if farm is TAP["env"].state[0].observation.farms[0] else 1
        fl[pl][item] += (1 if price > 1 else 0) if op == "SELL" else -1
    return ok


K._commit_unit = _tap


def extract_episode(row):
    streams = [OppEventStream(), OppEventStream()]      # stream[p] = events OF player p
    flows_out = [[], []]
    shed_out = [[], []]
    gen = ReplayDB.rebuild(None, row["episode_id"], row=row)
    _, env = next(gen)
    TAP["env"] = env
    step = 0
    while True:
        TAP["flows"] = [collections.Counter(), collections.Counter()]
        try:
            _, env = next(gen)
        except StopIteration:
            break
        fl = TAP["flows"]
        o0 = env.state[0].observation
        for p in (0, 1):
            priv = env.state[p].observation.private
            streams[p].push(step, fl[p], o0.farms[p], opp_money=o0.farms[p]["money"],
                            market_wheat=o0.market["inventory"]["WHEAT"])
            flows_out[p].append([fl[p][x] for x in PRODUCTS])
            shed_out[p].append([priv["shed"].get(x, 0) for x in PRODUCTS])
        step += 1
    TAP["flows"] = None
    out = []
    for k in (0, 1):
        o = 1 - k
        out.append(dict(episode_id=int(row["episode_id"]), seat=k, s=np.array(streams[o].s, np.int64),
                        d=np.array(streams[o].d, np.int64), flow=np.array(flows_out[o], np.int16),
                        shed=np.array(shed_out[o], np.int16)))
    return out


def _work(args):
    shard, out_path = args
    if os.path.exists(out_path):
        return out_path, 0
    import pyarrow.parquet as pq
    tab = pq.read_table(shard).to_pandas()
    trajs = []
    for _, r in tab.iterrows():
        try:
            trajs.extend(extract_episode(r.to_dict()))
        except Exception:
            continue
    if not trajs:
        return out_path, 0
    t_off = np.cumsum([0] + [len(t["s"]) for t in trajs]).astype(np.int32)
    d_off = np.cumsum([0] + [len(t["d"]) for t in trajs]).astype(np.int32)
    np.savez_compressed(out_path + ".tmp.npz", episode_id=np.array([t["episode_id"] for t in trajs], np.int64),
                        seat=np.array([t["seat"] for t in trajs], np.int8), t_off=t_off, d_off=d_off,
                        s_code=np.concatenate([t["s"] for t in trajs]), flow=np.concatenate([t["flow"] for t in trajs]),
                        opp_shed=np.concatenate([t["shed"] for t in trajs]),
                        d_code=np.concatenate([t["d"] for t in trajs]))
    os.replace(out_path + ".tmp.npz", out_path)
    return out_path, len(trajs)


def main():
    db = ReplayDB(sys.argv[1])
    out_dir = sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    shards = db.shards[-int(sys.argv[4]):] if len(sys.argv) > 4 else db.shards
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, os.path.join(out_dir, os.path.basename(s).replace(".parquet", ".npz"))) for s in shards]
    n = 0
    with mp.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        for path, k in pool.imap_unordered(_work, jobs):
            n += k
            print(f"{os.path.basename(path)} {k} trajs (total {n})", flush=True)


if __name__ == "__main__":
    main()
