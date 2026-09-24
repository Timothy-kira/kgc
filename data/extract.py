"""Replay DB -> training tensors (for representation / value / TTT pre-training on top play).

For every episode (both seats) re-simulate with the official interpreter, run the same
Tracker the agent uses, and store per step: product feats P, global feats G, the player's
actual sell bins (0..4, from its market orders), plus day targets and the final outcome.

python -m data.extract <db_dir> <out_dir> [min_score] [max_episodes] [procs]
Writes out_dir/part_XXXX.npz (one per chunk of episodes).
"""
import multiprocessing as mp
import os
import sys

import numpy as np

from agent.features import N_TGT, Tracker, label_bins
from data.replay_db import ReplayDB


def extract_row(r):
    trk = [Tracker(), Tracker()]
    rec = [dict(P=[], G=[], y=[]) for _ in range(2)]
    tg = [dict() for _ in range(2)]
    acts = r["actions"]
    if r["first_bad_step"] != -1:
        return None
    gen = ReplayDB.rebuild(None, r["episode_id"], row=r)
    for t, env in gen:
        if t >= len(acts) or env.done:
            break
        for p in (0, 1):
            o = env.obs(p, copy_obs=False)
            o = dict(o, step=t)
            trk[p].update(o)
            P, G, stock, _ = trk[p].features(None)
            a = acts[t][p] or {}
            rec[p]["P"].append(P)
            rec[p]["G"].append(G)
            rec[p]["y"].append(label_bins(a.get("market"), stock))
            for d, v in trk[p].pop_day_targets():
                tg[p][d] = v
    out = []
    for p in (0, 1):
        T = len(rec[p]["P"])
        nd = (T + 23) // 24
        day_tgt = np.zeros((nd, N_TGT), np.float32)
        day_mask = np.zeros(nd, np.float32)
        for d, v in tg[p].items():
            if d < nd:
                day_tgt[d], day_mask[d] = v, 1
        me, op = r[f"reward_{p}"], r[f"reward_{1 - p}"]
        out.append(dict(P=np.array(rec[p]["P"], np.float16), G=np.array(rec[p]["G"], np.float16),
                        y=np.array(rec[p]["y"], np.int8), day_tgt=day_tgt, day_mask=day_mask,
                        win=np.float32(1.0 if me > op else 0.5 if me == op else 0.0),
                        diff=np.float32((me - op) / 1e4), score=np.float32(r[f"updated_score_{p}"] or 0),
                        episode_id=np.int64(r["episode_id"]), seat=np.int8(p)))
    return out


def _work(args):
    db_dir, eids, path = args
    db = ReplayDB(db_dir)
    items = []
    for eid in eids:
        try:
            r = db.row(eid)
            from data.replay_db import _unz
            r["actions"] = _unz(r.pop("actions_zstd"))
            res = extract_row(r)
            if res:
                items.extend(res)
        except Exception as e:  # keep going on bad rows
            print("extract failed", eid, e, flush=True)
    if items:
        np.savez_compressed(path, **{f"{k}_{i}": v for i, it in enumerate(items) for k, v in it.items()},
                            n=len(items))
    return path, len(items)


def main():
    db_dir, out = sys.argv[1], sys.argv[2]
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else 2500
    mx = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    os.makedirs(out, exist_ok=True)
    db = ReplayDB(db_dir)
    df = db.episodes(min_score=ms)
    done = {f for f in os.listdir(out)}
    eids = df.sort_values("updated_score_0", ascending=False).episode_id.tolist()[:mx]
    chunks = [(db_dir, eids[i:i + 20], os.path.join(out, f"part_{i // 20:05d}.npz")) for i in range(0, len(eids), 20)]
    chunks = [c for c in chunks if os.path.basename(c[2]) not in done]
    with mp.Pool(procs) as p:
        for path, n in p.imap_unordered(_work, chunks):
            print(path, n, flush=True)


if __name__ == "__main__":
    main()
