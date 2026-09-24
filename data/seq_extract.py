"""Replay DB shard -> decoder-only training trajectories (observation arrays + action token ids).

For each episode (both seats) re-simulate with the official interpreter, build the per-step
observation tokens (agent/obs_tokens.py) and the exact action token ids (agent/action_tokens.py).
One output file per DB shard (incremental: finished shards are skipped).

python -m data.seq_extract <db_dir> <out_dir> [min_score] [procs]
Each output npz holds K trajectories concatenated:
    prod f16 [sumT,9,41]  glob f16 [sumT,16]  tiles u8 [sumT,2,100,22]  units f16 [sumT,15,35]
    act i16 [sumA]  act_off i32 [sumT + K]  (per trajectory: T+1 offsets, local)  t_off i32 [K+1]
    win f32 [K] diff f32 [K] score f32 [K] episode_id i64 [K] seat i8 [K]
"""
import multiprocessing as mp
import os
import sys

import numpy as np

from agent.action_tokens import encode
from agent.features import Tracker
from agent.obs_tokens import build_step
from data.replay_db import ReplayDB, _unz


def extract_episode(r):
    acts = r["actions"]
    trk = [Tracker(), Tracker()]
    rec = [dict(prod=[], glob=[], tiles=[], units=[], act=[], off=[0]) for _ in range(2)]
    for t, env in ReplayDB.rebuild(None, r["episode_id"], row=r):
        if t >= len(acts) or env.done:
            break
        for p in (0, 1):
            o = env.obs(p, copy_obs=False)
            trk[p].update(o)
            P, G, _, _ = trk[p].features(None)
            st = build_step(o, P, G)
            a = acts[t][p]
            ids = encode(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
            R = rec[p]
            R["prod"].append(st["prod"]); R["glob"].append(st["glob"])
            R["tiles"].append(st["tiles"]); R["units"].append(st["units"])
            R["act"].extend(ids); R["off"].append(len(R["act"]))
    out = []
    for p in (0, 1):
        me, op = r[f"reward_{p}"], r[f"reward_{1 - p}"]
        R = rec[p]
        out.append(dict(prod=np.array(R["prod"]), glob=np.array(R["glob"]), tiles=np.array(R["tiles"]),
                        units=np.array(R["units"]), act=np.array(R["act"], np.int16),
                        act_off=np.array(R["off"], np.int32),
                        win=np.float32(1.0 if me > op else 0.5 if me == op else 0.0),
                        diff=np.float32((me - op) / 1e4), score=np.float32(r[f"updated_score_{p}"] or 0),
                        episode_id=np.int64(r["episode_id"]), seat=np.int8(p)))
    return out


def save_trajs(path, trajs):
    cat = lambda k: np.concatenate([t[k] for t in trajs])
    t_off = np.cumsum([0] + [len(t["prod"]) for t in trajs]).astype(np.int32)
    a_off = np.cumsum([0] + [len(t["act"]) for t in trajs]).astype(np.int64)
    np.savez_compressed(path + ".tmp.npz", prod=cat("prod"), glob=cat("glob"), tiles=cat("tiles"), units=cat("units"),
                        act=cat("act"), act_off=cat("act_off"), t_off=t_off, a_off=a_off,
                        win=np.array([t["win"] for t in trajs]), diff=np.array([t["diff"] for t in trajs]),
                        score=np.array([t["score"] for t in trajs]),
                        episode_id=np.array([t["episode_id"] for t in trajs]),
                        seat=np.array([t["seat"] for t in trajs]))
    os.replace(path + ".tmp.npz", path)


def load_trajs(path):
    z = np.load(path)
    t_off, a_off = z["t_off"], z["a_off"]
    K = len(t_off) - 1
    ao = z["act_off"]
    out = []
    for i in range(K):
        s, e = t_off[i], t_off[i + 1]
        T = e - s
        out.append(dict(prod=z["prod"][s:e], glob=z["glob"][s:e], tiles=z["tiles"][s:e], units=z["units"][s:e],
                        act=z["act"][a_off[i]:a_off[i + 1]], act_off=ao[s + i: s + i + T + 1],
                        win=float(z["win"][i]), diff=float(z["diff"][i]), score=float(z["score"][i])))
    return out


def _work(args):
    shard, min_score, path = args
    import pyarrow.parquet as pq
    trajs = []
    for r in pq.read_table(shard).to_pylist():
        try:
            if min((r["updated_score_0"] or 0), (r["updated_score_1"] or 0)) < min_score or r["first_bad_step"] != -1:
                continue
            r["actions"] = _unz(r.pop("actions_zstd"))
            trajs.extend(extract_episode(r))
        except Exception as e:
            print("extract failed", r.get("episode_id"), repr(e), flush=True)
    if trajs:
        save_trajs(path, trajs)
    return path, len(trajs)


def main():
    db_dir, out = sys.argv[1], sys.argv[2]
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else 1800
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    shard_range = sys.argv[5] if len(sys.argv) > 5 else None     # "i:j" to split work across machines
    os.makedirs(out, exist_ok=True)
    db = ReplayDB(db_dir)
    shards = db.shards
    if shard_range:
        i, j = (int(x) for x in shard_range.split(":"))
        shards = shards[i:j]
    done = set(os.listdir(out))
    jobs = [(sh, ms, os.path.join(out, "seq_" + os.path.basename(sh).replace(".parquet", ".npz"))) for sh in shards]
    jobs = [j for j in jobs if os.path.basename(j[2]) not in done]
    with mp.Pool(procs) as p:
        for path, n in p.imap_unordered(_work, jobs):
            print(path, n, flush=True)


if __name__ == "__main__":
    main()
