"""Opponent mix shared by both RL tracks (docs/PLAN_RL.md, section 4).

Categories: near (ladder opponents rated 2200-2600, replayed open-loop from their tape: the market is the only
channel between players, so a tape is an exact opponent), loss (opponents that beat our submissions, same seed),
top (2700+), live (league agents that react to prices, incl. cha22) and self (v4b = cha22 + clamp_sells, the
exact mirror of our base). Tapes are exported once (`export`) into a small pickle with a held-out split by
episode id (every 5th episode -> eval), so kernels do not need the 1.7 GB replay DB.
"""
import copy
import json
import os
import pickle
import zlib

from agent.loader import call_adapter, entry_name, load_agent, load_module

LIVE = ["league/cha22.py", "league/metav4.py", "league/v48.py", "league/farm2945.py",
        "league/pub/hakfield/main.py", "league/pub/tetsutani_ms/main.py", "league/pub/pilkwang_sep/main.py",
        "league/pub/prvsiyan_frontier/main.py", "league/pub/dmitrii_2c1s/main.py", "league/pub/flexonafft_mpr/main.py"]
MIX = {"near": 0.30, "loss": 0.15, "top": 0.10, "live": 0.25, "self": 0.20}
EVAL_COUNTS = {"near": 18, "loss": 9, "top": 6, "live": 15, "self": 12}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class Tape1:
    """One seat's recorded actions, replayed open-loop."""

    def __init__(self, blob):
        self.acts = json.loads(zlib.decompress(blob))

    def __call__(self, obs, cfg=None):
        t = obs["step"]
        return copy.deepcopy(self.acts[t]) if t < len(self.acts) else PASS


class SeatTape(Tape1):
    """One seat of a full recorded episode (list of [a0, a1] per step)."""

    def __init__(self, acts, k):
        self.acts = [a[k] for a in acts]


def v4b():
    m = load_module("league/cha22.py")
    m._IMPL.chassis.cfg.update({"clamp_sells": True})
    return call_adapter(getattr(m, entry_name(m)))


def export(db_dir, out, n_near=800, n_top=400, loss_db=None):
    import tools.route_data as RD
    if loss_db:
        os.environ["ROUTE_LOSS_DB"] = loss_db
    pools = RD.load_mix(db_dir, n=n_near)
    pools["top"] = pools["top"][-n_top:]
    res = {"train": {}, "eval": {}}
    for cat, rows in pools.items():
        for eid, seed, cfg, acts, k in rows:
            blob = zlib.compress(json.dumps([a[k] for a in acts]).encode(), 6)
            cfg = {kk: v for kk, v in cfg.items() if v is not None}
            res["eval" if eid % 5 == 0 else "train"].setdefault(cat, []).append((eid, seed, cfg, blob, k))
    pickle.dump(res, open(out, "wb"))
    return {s: {c: len(v) for c, v in d.items()} for s, d in res.items()}


class OppMix:
    """Deterministic job id -> opponent spec over the category mix."""

    def __init__(self, pools, mix=MIX, seed_base=20000):
        self.pools = pools
        w = [(k, v) for k, v in mix.items() if v > 0 and (k in ("live", "self") or pools.get(k))]
        tot = sum(v for _, v in w)
        self.mix = [(k, v / tot) for k, v in w]
        self.seed_base = seed_base

    def category(self, j):
        u = ((j * 2654435761) % 2 ** 32) / 2 ** 32
        acc = 0.0
        for cat, wt in self.mix:
            acc += wt
            if u < acc:
                return cat
        return self.mix[-1][0]

    def spec(self, j, cat=None):
        """-> (kind, name, seed, seat, cfg, how) with how = ("tape", blob) | ("agent", path) | ("v4b",)."""
        cat = cat or self.category(j)
        if cat in ("live", "self"):
            seed, seat = self.seed_base + j, j % 2
            if cat == "self":
                return "self", "v4b", seed, seat, None, ("v4b",)
            path = LIVE[(j // 2) % len(LIVE)]
            return "live", path, seed, seat, None, ("agent", path)
        pool = self.pools[cat]
        eid, seed, cfg, blob, k = pool[(j * 7919) % len(pool)]
        return cat, f"{cat}:{eid}", seed, 1 - k, cfg, ("tape", blob)

    @staticmethod
    def make(how):
        if how[0] == "tape":
            return Tape1(how[1])
        if how[0] == "v4b":
            return v4b()
        return load_agent(how[1])

    def eval_specs(self, counts=EVAL_COUNTS, start=10 ** 6):
        """Stratified, fixed evaluation jobs (use with the held-out pools)."""
        out = []
        for cat, n in counts.items():
            if cat not in ("live", "self") and not self.pools.get(cat):
                continue
            for i in range(n):
                j = start + len(out)
                out.append((j, self.spec(j, cat)))
        return out


def load_pools(path, split="train"):
    return pickle.load(open(path, "rb"))[split]


if __name__ == "__main__":
    import sys
    print(export(sys.argv[1], sys.argv[2], loss_db=sys.argv[3] if len(sys.argv) > 3 else None))


def export_v2(db_dir, out, caps=None, loss_db=None, demo_min=2600, eval_caps=None):
    """B-track pools from the whole replay DB (docs/PLAN_RL.md section 8): every seat of every episode is a
    candidate opponent tape, categorised by that player's rating (top >= 2700, near 2200-2700); loss = opponents
    that beat our submissions; demo = full episodes whose winner is rated >= demo_min (both seats kept, replayed
    from the winner's seat as offline imitation data). Most recent episodes first; every 5th episode id -> eval."""
    import numpy as np
    import pyarrow.parquet as pq
    from data.replay_db import ReplayDB, _unz
    caps = caps or {"near": 15000, "top": 8000, "loss": 100000, "demo": 6000}
    eval_caps = eval_caps or {"near": 400, "top": 200, "loss": 1000}
    OUR = {56553173, 56564007}
    want = {}                                              # eid -> list of (split, cat, seat)
    counts = {}
    for root in [db_dir] + ([loss_db] if loss_db else []):
        e = ReplayDB(root).episodes().sort_values("episode_id", ascending=False)
        for r in e.itertuples():
            eid = int(r.episode_id)
            split = "eval" if eid % 5 == 0 else "train"
            s = [r.updated_score_0 or 0, r.updated_score_1 or 0]
            rw = [r.reward_0 or 0, r.reward_1 or 0]
            subs = [r.submission_id_0, r.submission_id_1]
            items = []
            for k in (0, 1):
                ours = subs[1 - k] in OUR
                if ours and rw[k] >= rw[1 - k]:
                    items.append((split, "loss", k))
                elif not ours and subs[k] not in OUR:
                    cat = "top" if s[k] >= 2700 else ("near" if s[k] >= 2200 else None)
                    if cat:
                        items.append((split, cat, k))
            w = 0 if rw[0] > rw[1] else (1 if rw[1] > rw[0] else None)
            if split == "train" and w is not None and s[w] >= demo_min and subs[w] not in OUR:
                items.append((split, "demo", w))
            keep = []
            for split_, cat, k in items:
                cap = (caps if split_ == "train" else eval_caps).get(cat, 0)
                key = (split_, cat)
                if counts.get(key, 0) < cap:
                    counts[key] = counts.get(key, 0) + 1
                    keep.append((split_, cat, k))
            if keep:
                want.setdefault((root, eid), []).extend(keep)
    res = {"train": {}, "eval": {}}
    by_root = {}
    for (root, eid), v in want.items():
        by_root.setdefault(root, {})[eid] = v
    for root, eids in by_root.items():
        for shard in ReplayDB(root).shards:
            t = pq.read_table(shard, columns=["episode_id", "seed", "config", "actions_zstd"]).to_pylist()
            for row in t:
                eid = int(row["episode_id"])
                if eid not in eids:
                    continue
                acts = _unz(row["actions_zstd"])
                cfg = {kk: vv for kk, vv in json.loads(row["config"]).items() if vv is not None}
                for split_, cat, k in eids.pop(eid):
                    if cat == "demo":
                        blob = zlib.compress(json.dumps(acts).encode(), 6)
                    else:
                        blob = zlib.compress(json.dumps([a[k] for a in acts]).encode(), 6)
                    res[split_].setdefault(cat, []).append((eid, int(row["seed"]), cfg, blob, k))
    pickle.dump(res, open(out, "wb"))
    return {s: {c: len(v) for c, v in d.items()} for s, d in res.items()}
