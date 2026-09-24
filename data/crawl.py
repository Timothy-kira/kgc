"""Resumable large-scale Kaggriculture replay crawler -> compact replay database.

Design
- Discovery: best-first BFS over the public episode graph. Listing a submission's
  episodes reveals its opponents' submission ids and ratings, which are pushed
  onto the frontier (highest rating first).
- Download: each replay (~33 MB JSON) is streamed, compacted to
  seed + both players' per-step actions + light metadata (~50-150 KB zstd),
  and the raw JSON is dropped. The environment is deterministic, so
  (seed, actions) reproduces every state exactly (see env/replay_check.py).
- Resume: all progress lives in OUT_DIR (atomic writes). Re-running continues
  from the last checkpoint; if a previous run's output is mounted read-only
  (e.g. a Kaggle notebook's own previous version), pass it as --resume-from and
  it is copied into OUT_DIR first. Already-downloaded episodes are never
  fetched again, so a re-run only picks up new games.

Layout of OUT_DIR
    state.json                     frontier / visited submissions / counters
    index/episodes_index.parquet   every discovered episode (metadata only)
    index/teams.parquet            teamId -> team name
    shards/ep_XXXXX.parquet        downloaded episodes (one row per episode)
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import os
import random
import shutil
import threading
import time
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{}.json"
UA = {"User-Agent": "kaggriculture-replay-db/1.0", "Content-Type": "application/json"}

CFG_KEYS = ("episodeSteps", "boardSize", "startingMoney", "maxMarketOrdersPerTurn", "turnsPerDay",
            "shedCapacity", "weedSpawnChance", "townShopUnlockInterval", "townShopSellInterval",
            "townCenterSellInterval", "farmHandCostMult", "marketParams")


# --------------------------------------------------------------------------- io helpers
def _http(req_fn, tries=8):
    delay = 2.0
    for k in range(tries):
        try:
            return req_fn()
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404):
                raise
            if e.code == 429:  # rate limited: back off hard
                delay = max(delay, 15.0)
            err = e
        except Exception as e:  # network hiccup
            err = e
        time.sleep(delay + random.random())
        delay = min(delay * 2, 180)
    raise err


def list_episodes(submission_id):
    body = json.dumps({"submissionId": int(submission_id)}).encode()

    def go():
        req = urllib.request.Request(LIST_URL, data=body, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    return _http(go)


def fetch_replay(episode_id):
    def go():
        req = urllib.request.Request(REPLAY_URL.format(episode_id), headers={"User-Agent": UA["User-Agent"]})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    return _http(go)


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_write_table(path, table):
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- compaction
def _zc(data):
    # ZstdCompressor objects are not thread-safe: one per call.
    return zstd.ZstdCompressor(level=10).compress(data)


def compact_replay(rep, meta):
    steps = rep["steps"]
    acts = [[steps[t + 1][0].get("action"), steps[t + 1][1].get("action")] for t in range(len(steps) - 1)]
    statuses = [[s.get("status") for s in st] for st in steps]
    # First step index where any agent stopped being ACTIVE/DONE-normal (errors, timeouts).
    bad = [t for t, st in enumerate(statuses) if any(x in ("ERROR", "TIMEOUT", "INVALID") for x in st)]
    daily = []
    tpd = rep["configuration"].get("turnsPerDay", 24)
    for t in range(0, len(steps), tpd):
        o = steps[t][0]["observation"]
        daily.append({
            "step": t,
            "money": [f["money"] for f in o["farms"]],
            "quadrants": [len(f["unlocked_quadrants"]) for f in o["farms"]],
            "prices": o["market"]["prices"],
            "inventory": o["market"]["inventory"],
            "shops": o["town"]["unlocked_shops"],
        })
    rewards = [s.get("reward") for s in steps[-1]]
    info = rep.get("info", {})
    agents = sorted(meta["agents"], key=lambda a: a.get("index", 0))
    return {
        "episode_id": int(meta["id"]),
        "create_time": meta.get("createTime"),
        "seed": int(info.get("seed")) if info.get("seed") is not None else None,
        "team_names": json.dumps(info.get("TeamNames")),
        "team_id_0": agents[0].get("teamId"), "team_id_1": agents[1].get("teamId"),
        "submission_id_0": agents[0].get("submissionId"), "submission_id_1": agents[1].get("submissionId"),
        "initial_score_0": agents[0].get("initialScore"), "initial_score_1": agents[1].get("initialScore"),
        "updated_score_0": agents[0].get("updatedScore"), "updated_score_1": agents[1].get("updatedScore"),
        "reward_0": rewards[0], "reward_1": rewards[1],
        "n_steps": len(steps),
        "first_bad_step": bad[0] if bad else -1,
        "final_status": json.dumps(statuses[-1]),
        "config": json.dumps({k: rep["configuration"].get(k) for k in CFG_KEYS}),
        "actions_zstd": _zc(json.dumps(acts, separators=(",", ":")).encode()),
        "daily_zstd": _zc(json.dumps(daily, separators=(",", ":")).encode()),
    }


EP_SCHEMA = pa.schema([
    ("episode_id", pa.int64()), ("create_time", pa.string()), ("seed", pa.int64()), ("team_names", pa.string()),
    ("team_id_0", pa.int64()), ("team_id_1", pa.int64()),
    ("submission_id_0", pa.int64()), ("submission_id_1", pa.int64()),
    ("initial_score_0", pa.float64()), ("initial_score_1", pa.float64()),
    ("updated_score_0", pa.float64()), ("updated_score_1", pa.float64()),
    ("reward_0", pa.float64()), ("reward_1", pa.float64()),
    ("n_steps", pa.int32()), ("first_bad_step", pa.int32()), ("final_status", pa.string()),
    ("config", pa.string()), ("actions_zstd", pa.binary()), ("daily_zstd", pa.binary()),
])


# --------------------------------------------------------------------------- crawler
class Crawler:
    def __init__(self, out_dir, min_score, max_hours, workers, list_workers, shard_size, max_episodes,
                 verify_fraction):
        self.out = out_dir
        self.min_score = min_score
        self.deadline = time.time() + max_hours * 3600
        self.workers, self.list_workers = workers, list_workers
        self.shard_size = shard_size
        self.max_episodes = max_episodes
        self.verify_fraction = verify_fraction
        os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "index"), exist_ok=True)
        self.lock = threading.Lock()
        self._load()

    # ---- persistence
    def _load(self):
        p = os.path.join(self.out, "state.json")
        st = json.load(open(p)) if os.path.exists(p) else {}
        self.frontier = {int(k): v for k, v in st.get("frontier", {}).items()}   # sub -> score
        self.sub_listed_at = {int(k): v for k, v in st.get("sub_listed_at", {}).items()}
        self.shard_no = st.get("shard_no", 0)
        self.index = {}   # episode_id -> meta dict
        ip = os.path.join(self.out, "index", "episodes_index.parquet")
        if os.path.exists(ip):
            for r in pq.read_table(ip).to_pylist():
                self.index[r["episode_id"]] = r
        self.teams = {}
        tp = os.path.join(self.out, "index", "teams.parquet")
        if os.path.exists(tp):
            self.teams = {r["team_id"]: r["team_name"] for r in pq.read_table(tp).to_pylist()}
        self.done = set()
        for f in glob.glob(os.path.join(self.out, "shards", "ep_*.parquet")):
            self.done.update(pq.read_table(f, columns=["episode_id"]).column(0).to_pylist())
        self.failed = set(st.get("failed", []))
        self.list_fail = {}
        self.buffer = []
        print(f"[resume] frontier={len(self.frontier)} listed_subs={len(self.sub_listed_at)} "
              f"indexed_eps={len(self.index)} downloaded={len(self.done)} shards={self.shard_no}", flush=True)

    def save(self):
        with self.lock:
            self._flush_buffer(force=True)
            _atomic_write_json(os.path.join(self.out, "state.json"), {
                "frontier": self.frontier, "sub_listed_at": self.sub_listed_at, "shard_no": self.shard_no,
                "failed": sorted(self.failed), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_downloaded": len(self.done), "n_indexed": len(self.index),
            })
            rows = list(self.index.values())
            if rows:
                _atomic_write_table(os.path.join(self.out, "index", "episodes_index.parquet"), pa.Table.from_pylist(rows))
            if self.teams:
                _atomic_write_table(os.path.join(self.out, "index", "teams.parquet"),
                                    pa.Table.from_pylist([{"team_id": k, "team_name": v} for k, v in self.teams.items()]))

    def _flush_buffer(self, force=False):
        if not self.buffer or (len(self.buffer) < self.shard_size and not force):
            return
        path = os.path.join(self.out, "shards", f"ep_{self.shard_no:05d}.parquet")
        _atomic_write_table(path, pa.Table.from_pylist(self.buffer, schema=EP_SCHEMA))
        self.shard_no += 1
        self.buffer = []

    # ---- discovery
    def add_seeds(self, subs):
        for s in subs:
            self.frontier.setdefault(int(s), 1e9)

    def _record_listing(self, sub, resp):
        teams = {t["id"]: t.get("teamName") for t in resp.get("teams", []) if "id" in t}
        with self.lock:
            self.teams.update({k: v for k, v in teams.items() if v})
            for e in resp.get("episodes", []):
                if e.get("state") != "COMPLETED" or len(e.get("agents", [])) != 2:
                    continue
                ag = sorted(e["agents"], key=lambda a: a.get("index", 0))
                sc = [a.get("updatedScore") or a.get("initialScore") or 0 for a in ag]
                self.index[e["id"]] = {
                    "episode_id": int(e["id"]), "create_time": e.get("createTime"), "type": e.get("type"),
                    "submission_id_0": ag[0].get("submissionId"), "submission_id_1": ag[1].get("submissionId"),
                    "team_id_0": ag[0].get("teamId"), "team_id_1": ag[1].get("teamId"),
                    "score_0": float(sc[0]), "score_1": float(sc[1]),
                    "reward_0": ag[0].get("reward"), "reward_1": ag[1].get("reward"),
                }
                for a, s in zip(ag, sc):
                    sid = a.get("submissionId")
                    if sid is None:
                        continue
                    if s >= self.min_score and (sid not in self.sub_listed_at):
                        self.frontier[sid] = max(self.frontier.get(sid, 0), s)
            self.sub_listed_at[sub] = time.time()
            self.frontier.pop(sub, None)

    def discover(self, max_lists):
        """Best-first expansion of the submission frontier."""
        n = 0
        with cf.ThreadPoolExecutor(self.list_workers) as ex:
            while n < max_lists and time.time() < self.deadline:
                with self.lock:
                    batch = sorted(self.frontier.items(), key=lambda kv: -kv[1])[: self.list_workers * 2]
                if not batch:
                    break
                futs = {ex.submit(list_episodes, s): s for s, _ in batch}
                for f in cf.as_completed(futs):
                    s = futs[f]
                    try:
                        self._record_listing(s, f.result())
                    except Exception as e:
                        print(f"[list] sub {s} failed: {e}", flush=True)
                        with self.lock:
                            # keep it on the frontier (lower priority) unless it keeps failing
                            self.list_fail[s] = self.list_fail.get(s, 0) + 1
                            if self.list_fail[s] >= 3:
                                self.frontier.pop(s, None)
                                self.sub_listed_at[s] = time.time()
                            else:
                                self.frontier[s] = self.frontier.get(s, 0) - 500
                        time.sleep(10)
                    n += 1
                if n % 50 < len(batch):
                    print(f"[discover] listed={len(self.sub_listed_at)} frontier={len(self.frontier)} "
                          f"episodes={len(self.index)}", flush=True)
                    self.save()
        self.save()

    def refresh_listings(self, min_age_hours=6.0):
        """On resume: re-list known high-rated submissions to pick up new games."""
        now = time.time()
        with self.lock:
            best = {}
            for m in self.index.values():
                for k in (0, 1):
                    sid, sc = m[f"submission_id_{k}"], m[f"score_{k}"]
                    if sid is not None and sc >= self.min_score:
                        best[sid] = max(best.get(sid, 0), sc)
            for sid, sc in best.items():
                if now - self.sub_listed_at.get(sid, 0) > min_age_hours * 3600:
                    self.sub_listed_at.pop(sid, None)
                    self.frontier[sid] = max(self.frontier.get(sid, 0), sc)

    # ---- download
    def _pending(self):
        with self.lock:
            cands = [m for eid, m in self.index.items()
                     if eid not in self.done and eid not in self.failed
                     and min(m["score_0"], m["score_1"]) >= self.min_score]
        cands.sort(key=lambda m: -min(m["score_0"], m["score_1"]))  # strongest games first
        return cands

    def _download_one(self, meta):
        rep = fetch_replay(meta["episode_id"])
        full = {"id": meta["episode_id"], "createTime": meta["create_time"], "agents": [
            {"index": k, "submissionId": meta[f"submission_id_{k}"], "teamId": meta[f"team_id_{k}"],
             "updatedScore": meta[f"score_{k}"], "initialScore": None} for k in (0, 1)]}
        row = compact_replay(rep, full)
        if self.verify_fraction > 0 and random.random() < self.verify_fraction:
            from env.replay_check import check
            ok, _ = check(rep, verbose=False)
            if not ok:
                raise RuntimeError(f"replay {meta['episode_id']} failed local re-simulation")
        del rep
        return row

    def download(self):
        pending = self._pending()
        if self.max_episodes:
            pending = pending[: max(0, self.max_episodes - len(self.done))]
        print(f"[download] pending={len(pending)}", flush=True)
        t0, got = time.time(), 0
        with cf.ThreadPoolExecutor(self.workers) as ex:
            it = iter(pending)
            futs = {}
            for m in it:
                futs[ex.submit(self._download_one, m)] = m
                if len(futs) >= self.workers * 2:
                    break
            while futs:
                done, _ = cf.wait(futs, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    m = futs.pop(f)
                    try:
                        row = f.result()
                        with self.lock:
                            self.buffer.append(row)
                            self.done.add(m["episode_id"])
                            self._flush_buffer()
                        got += 1
                    except Exception as e:
                        print(f"[download] ep {m['episode_id']} failed: {e}", flush=True)
                        with self.lock:
                            self.failed.add(m["episode_id"])
                    if got and got % 100 == 0:
                        rate = got / (time.time() - t0)
                        print(f"[download] {got} new ({len(self.done)} total) {rate:.2f} ep/s", flush=True)
                        self.save()
                if time.time() < self.deadline:
                    for m in it:
                        futs[ex.submit(self._download_one, m)] = m
                        if len(futs) >= self.workers * 2:
                            break
        self.save()
        return got


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="replay_db")
    ap.add_argument("--resume-from", default=None, help="read-only previous DB to continue from")
    ap.add_argument("--seed-subs", default="", help="comma-separated submission ids to start from")
    ap.add_argument("--min-score", type=float, default=1500.0)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--max-lists", type=int, default=100000)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list-workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--verify-fraction", type=float, default=0.0)
    ap.add_argument("--rounds", type=int, default=3, help="discover/download alternations")
    a = ap.parse_args(argv)

    if a.resume_from and os.path.isdir(a.resume_from) and not os.path.exists(os.path.join(a.out, "state.json")):
        print(f"[resume] copying {a.resume_from} -> {a.out}", flush=True)
        shutil.copytree(a.resume_from, a.out, dirs_exist_ok=True)

    c = Crawler(a.out, a.min_score, a.max_hours, a.workers, a.list_workers, a.shard_size, a.max_episodes,
                a.verify_fraction)
    c.add_seeds([s for s in a.seed_subs.split(",") if s.strip()])
    c.refresh_listings()
    for r in range(a.rounds):
        if time.time() >= c.deadline:
            break
        c.download()          # fetch everything already discovered first (listing is rate limited)
        c.discover(a.max_lists)
    c.save()
    print(f"[done] downloaded={len(c.done)} indexed={len(c.index)} shards={c.shard_no}", flush=True)


if __name__ == "__main__":
    main()
