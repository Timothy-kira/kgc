"""Official daily episode datasets (kaggle/kaggriculture-episodes-YYYY-MM-DD) -> replay-DB shards.

python -m data.import_official <out_dir> <base_db> <json_glob> [procs=4] [verify_fraction=0.01] [manifest.csv]
The official replays carry no submission ids / ratings, only team names: team_id comes from the base DB's
index/teams.parquet and updated_score_k from that team's latest rating in the base DB (fallback: the day's
median_avg_score from the episodes-index manifest). Episodes already in the base DB shards are skipped.
Output: <out_dir>/shards/ep_off_<YYYYMMDD>_<NNN>.parquet (same EP_SCHEMA as data/crawl.py, so ReplayDB, seq and
opponent-event extraction work unchanged on a merged shard list).
"""
import collections
import glob
import json
import multiprocessing as mp
import os
import random
import re
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.crawl import EP_SCHEMA, _atomic_write_table, compact_replay

SHARD = 250
G = {}


def _day(path):
    m = re.search(r"episodes-(\d{4})-(\d{2})-(\d{2})", path)
    return "".join(m.groups()) if m else "00000000"


def _work(path):
    try:
        rep = json.load(open(path))
        eid = int(rep["info"]["EpisodeId"])
        if eid in G["have"]:
            return path, None, "have"
        day = _day(path)
        agents = []
        for k, name in enumerate(rep["info"].get("TeamNames") or [None, None]):
            tid = G["team_id"].get(name)
            sc = G["team_score"].get(tid, G["day_score"].get(day))
            agents.append({"index": k, "teamId": tid, "submissionId": None, "updatedScore": sc, "initialScore": None})
        meta = {"id": eid, "createTime": f"{day[:4]}-{day[4:6]}-{day[6:]}T12:00:00Z", "agents": agents}
        row = compact_replay(rep, meta)
        if G["verify"] > 0 and random.random() < G["verify"]:
            from env.replay_check import check
            ok, _ = check(rep, verbose=False)
            if not ok:
                return path, None, "verify_failed"
        return path, row, "ok"
    except Exception as e:                      # truncated / odd file: skip, keep going
        return path, None, f"error {type(e).__name__}: {e}"


def main():
    out, base, pattern = sys.argv[1:4]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    G["verify"] = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
    manifest = sys.argv[6] if len(sys.argv) > 6 else None
    have = set()
    for s in glob.glob(os.path.join(base, "shards", "ep_*.parquet")):
        have.update(pq.read_table(s, columns=["episode_id"]).column(0).to_pylist())
    G["have"] = have
    teams = pd.read_parquet(os.path.join(base, "index", "teams.parquet"))
    G["team_id"] = dict(zip(teams.team_name, teams.team_id.astype(int)))
    idx = pd.read_parquet(os.path.join(base, "index", "episodes_index.parquet")).sort_values("create_time")
    ts = {}
    for k in (0, 1):
        for tid, sc in zip(idx[f"team_id_{k}"], idx[f"score_{k}"]):
            if tid is not None and sc == sc:
                ts[int(tid)] = float(sc)       # sorted by time: last write = latest rating
    G["team_score"] = ts
    G["day_score"] = {}
    if manifest and os.path.exists(manifest):
        m = pd.read_csv(manifest)
        G["day_score"] = {d.replace("-", ""): float(s) for d, s in zip(m.date, m.median_avg_score)}
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"[official] files={len(files)} base_eps={len(have)} teams={len(ts)}", flush=True)
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    buf, shard_no, stats = collections.defaultdict(list), collections.Counter(), collections.Counter()

    def flush(day, force=False):
        while len(buf[day]) >= SHARD or (force and buf[day]):
            rows, buf[day] = buf[day][:SHARD], buf[day][SHARD:]
            p = os.path.join(out, "shards", f"ep_off_{day}_{shard_no[day]:03d}.parquet")
            _atomic_write_table(p, pa.Table.from_pylist(rows, schema=EP_SCHEMA))
            shard_no[day] += 1

    with mp.get_context("fork").Pool(procs, maxtasksperchild=50) as pool:
        for i, (path, row, st) in enumerate(pool.imap_unordered(_work, files, chunksize=2)):
            stats[st.split(" ")[0]] += 1
            if st.startswith("error") or st == "verify_failed":
                print(f"[official] {os.path.basename(path)} {st}", flush=True)
            if row is not None:
                d = _day(path)
                buf[d].append(row)
                flush(d)
            if (i + 1) % 500 == 0:
                print(f"[official] {i + 1}/{len(files)} {dict(stats)}", flush=True)
    for d in list(buf):
        flush(d, force=True)
    print(f"[official] done {dict(stats)} shards={sum(shard_no.values())}", flush=True)


if __name__ == "__main__":
    main()
