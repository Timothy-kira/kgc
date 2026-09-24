"""Reader for the compact Kaggriculture replay database produced by data/crawl.py.

    from data.replay_db import ReplayDB
    db = ReplayDB("/kaggle/input/kaggriculture-replay-db/replay_db")
    df = db.episodes(min_score=2800)            # pandas DataFrame of metadata
    acts = db.actions(df.episode_id.iloc[0])   # list[[a0, a1]] per transition
    for t, env in db.rebuild(eid): ...          # exact re-simulation (needs kaggle_environments)
"""
import glob
import json
import os

import pyarrow.parquet as pq
import zstandard as zstd


def _unz(b):
    return json.loads(zstd.ZstdDecompressor().decompress(b))


class ReplayDB:
    def __init__(self, root):
        self.root = root
        self.shards = sorted(glob.glob(os.path.join(root, "shards", "ep_*.parquet")))
        self._loc = None

    def episodes(self, min_score=None, columns=None):
        import pandas as pd
        meta_cols = [c for c in pq.read_schema(self.shards[0]).names if c not in ("actions_zstd", "daily_zstd")]
        frames = []
        for s in self.shards:
            df = pq.read_table(s, columns=columns or meta_cols).to_pandas()
            df["shard"] = s
            frames.append(df)
        df = pd.concat(frames, ignore_index=True).drop_duplicates("episode_id")
        if min_score is not None:
            df = df[df[["updated_score_0", "updated_score_1"]].min(axis=1) >= min_score]
        return df.reset_index(drop=True)

    def _locate(self):
        if self._loc is None:
            self._loc = {}
            for s in self.shards:
                for eid in pq.read_table(s, columns=["episode_id"]).column(0).to_pylist():
                    self._loc[eid] = s
        return self._loc

    def row(self, episode_id):
        s = self._locate()[episode_id]
        t = pq.read_table(s, filters=[("episode_id", "=", episode_id)])
        return t.to_pylist()[0]

    def iter_rows(self, min_score=None):
        """Stream full rows (with decoded actions/daily) shard by shard."""
        for s in self.shards:
            for r in pq.read_table(s).to_pylist():
                if min_score is not None and min(r["updated_score_0"] or 0, r["updated_score_1"] or 0) < min_score:
                    continue
                r["actions"] = _unz(r.pop("actions_zstd"))
                r["daily"] = _unz(r.pop("daily_zstd"))
                yield r

    def actions(self, episode_id):
        return _unz(self.row(episode_id)["actions_zstd"])

    def daily(self, episode_id):
        return _unz(self.row(episode_id)["daily_zstd"])

    def rebuild(self, episode_id, row=None):
        """Yield (t, env) after each transition, re-simulated with the official interpreter."""
        from env.fast_env import FarmEnv
        r = row or self.row(episode_id)
        acts = r["actions"] if "actions" in r else _unz(r["actions_zstd"])
        cfg = {k: v for k, v in json.loads(r["config"]).items() if v is not None}
        env = FarmEnv(r["seed"], cfg)
        yield 0, env
        for t, (a0, a1) in enumerate(acts):
            if env.done:
                break
            env.step(a0, a1)
            yield t + 1, env
