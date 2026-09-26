"""Download our own ladder games (official kaggle CLI) into a compact replay DB, for loss analysis / loss pool.

python -m tools.fetch_our_games <out_db_dir> <submission_id,...> [team_name="xisheng feng"]
Uses `kaggle competitions episodes <sub> --format json` + `kaggle competitions replay <episode>` (the internal
listing API used by data/crawl.py is rate limited while the crawler runs). Each replay (~35 MB) is compacted with
data.crawl.compact_replay into data/replay_db format and deleted; resumable (existing episode ids skipped).
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from data.crawl import EP_SCHEMA, compact_replay


def episodes(sub):
    out = subprocess.run(["kaggle", "competitions", "episodes", str(sub), "--format", "json"], capture_output=True,
                         text=True, timeout=300).stdout
    j = out[out.index("["):out.rindex("]") + 1]
    return [e for e in json.loads(j) if "COMPLETED" in e.get("state", "")]


def main():
    out, subs = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    team = sys.argv[3] if len(sys.argv) > 3 else "xisheng feng"
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    have = set()
    for f in glob.glob(os.path.join(out, "shards", "*.parquet")):
        have |= set(pq.read_table(f, columns=["episode_id"]).column(0).to_pylist())
    rows = []
    tmp = tempfile.mkdtemp()
    for sub in subs:
        eps = episodes(sub)
        print(f"sub {sub}: {len(eps)} completed episodes", flush=True)
        for e in eps:
            if e["id"] in have:
                continue
            subprocess.run(["kaggle", "competitions", "replay", str(e["id"]), "-p", tmp], capture_output=True, timeout=600)
            fs = glob.glob(os.path.join(tmp, f"*{e['id']}*.json"))
            if not fs:
                continue
            rep = json.load(open(fs[0]))
            os.remove(fs[0])
            names = rep.get("info", {}).get("TeamNames") or ["", ""]
            me = names.index(team) if team in names else 0
            meta = {"id": e["id"], "createTime": e.get("createTime"), "agents": [
                {"index": k, "submissionId": sub if k == me else None, "teamId": None, "updatedScore": None,
                 "initialScore": None} for k in (0, 1)]}
            rows.append(compact_replay(rep, meta))
            have.add(e["id"])
            r = rows[-1]
            print(json.dumps({"episode": e["id"], "sub": sub, "opp": names[1 - me],
                              "diff": (r[f"reward_{me}"] or 0) - (r[f"reward_{1 - me}"] or 0)}), flush=True)
            if len(rows) >= 25:
                n = len(glob.glob(os.path.join(out, "shards", "*.parquet")))
                pq.write_table(pa.Table.from_pylist(rows, schema=EP_SCHEMA), os.path.join(out, "shards", f"ep_{n:05d}.parquet"))
                rows = []
    if rows:
        n = len(glob.glob(os.path.join(out, "shards", "*.parquet")))
        pq.write_table(pa.Table.from_pylist(rows, schema=EP_SCHEMA), os.path.join(out, "shards", f"ep_{n:05d}.parquet"))


if __name__ == "__main__":
    main()
