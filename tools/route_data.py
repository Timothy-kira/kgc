"""H5 training data: value of every cha22 route at step 144, per game (docs/PLAN_v4.1.md, Engram replaces cha22's
route tables).

python -m tools.route_data <out.jsonl> <jobs_from> <jobs_to> [procs] [db_dir]
Jobs are deterministic (index -> opponent, seed, seat): opponents cycle over live league agents and, when a replay
DB is given, top-team replays whose opponent seat is replayed open-loop (the market is the only channel between
players, so a tape is an exact opponent for us). Our side is cha22 up to step 144; then the game forks over all
routes (shared prefix 0-143, terminal route 2 from 648 as cha22 does) and each branch plays to the end.
Each output line: job, opponent, seed, seat, first two shops, opponent D codes and S codes up to step 144
(agent/opp_events.py), cha22's own route, and {route: [our money, opponent money]}.
"""
import collections
import copy
import importlib
import json
import os
import sys

from agent.loader import call_adapter, entry_name, load_agent, load_module
from agent.opp_events import PRODUCTS, OppEventStream
from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
CHA22 = "league/cha22.py"
LIVE = ["league/cha22.py", "league/metav4.py", "league/v48.py", "league/farm2945.py",
        "league/pub/hakfield/main.py", "league/pub/tetsutani_ms/main.py", "league/pub/pilkwang_sep/main.py",
        "league/pub/prvsiyan_frontier/main.py", "league/pub/dmitrii_2c1s/main.py", "league/pub/flexonafft_mpr/main.py"]
DECISION = 144

_orig = K._commit_unit
TAP = {"env": None, "fl": None}


def _find_farm(args, kw):
    for v in list(args) + list(kw.values()):
        if isinstance(v, dict) and "money" in v and "tiles" in v:
            return v
    return None


def _tap(*args, **kw):
    """Signature-agnostic tap (Kaggle's kaggle_environments added an argument to _commit_unit)."""
    ok = _orig_commit(*args, **kw) if "_orig_commit" in globals() else _orig(*args, **kw)
    op, item, price = (list(args) + [None] * 3)[:3]
    op, item, price = kw.get("op", op), kw.get("item", item), kw.get("price", price)
    fl = TAP["fl"]
    if ok and fl is not None and op in ("SELL", "BUY_PRODUCT"):
        farm = _find_farm(args, kw)
        pl = 0 if farm is TAP["env"].state[0].observation.farms[0] else 1
        fl[pl][item] += (1 if (price or 0) > 1 else 0) if op == "SELL" else -1
    return ok


K._commit_unit = _tap


class Tape:
    def __init__(self, actions, seat):
        self.acts, self.seat = actions, seat

    def __call__(self, obs, cfg=None):
        t = obs["step"]
        return copy.deepcopy(self.acts[t][self.seat]) if t < len(self.acts) else {"farmer": ["PASS"], "hands": [], "market": []}


TAPES = []


def load_tapes(db_dir, n=400, min_score=2700):
    from data.replay_db import ReplayDB, _unz
    db = ReplayDB(db_dir)
    e = db.episodes()
    e = e[e[["updated_score_0", "updated_score_1"]].max(axis=1) >= min_score].sort_values("episode_id").tail(n)
    out = []
    for r in e.itertuples():
        row = db.row(int(r.episode_id))
        k = 0 if (r.updated_score_0 or 0) >= (r.updated_score_1 or 0) else 1       # the stronger player is the opponent
        out.append((int(r.episode_id), int(r.seed), json.loads(row["config"]), _unz(row["actions_zstd"]), k))
    return out


def job_spec(j):
    """Deterministic job -> (kind, opponent, seed, seat, cfg, tape)."""
    seat = j % 2
    if TAPES and j % 3 == 2:
        eid, seed, cfg, acts, k = TAPES[(j // 3) % len(TAPES)]
        return "tape", f"replay:{eid}", seed, 1 - k, {kk: v for kk, v in cfg.items() if v is not None}, Tape(acts, k)
    opp = LIVE[(j // 2) % len(LIVE)]
    return "live", opp, 20000 + j, seat, None, None


def run_job(j):
    best_effort()
    kind, opp_name, seed, seat, cfg, tape = job_spec(j)
    m = load_module(CHA22)
    me = call_adapter(getattr(m, entry_name(m)))
    opp = tape if tape is not None else load_agent(opp_name)
    env = FarmEnv(seed, cfg)
    TAP["env"] = env
    stream = OppEventStream()
    while env.step_count < DECISION and not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        TAP["fl"] = [collections.Counter(), collections.Counter()]
        t = env.step_count
        env.step(*((a, b) if seat == 0 else (b, a)))
        o0 = env.state[0].observation
        stream.push(t, dict(TAP["fl"][1 - seat]), o0.farms[1 - seat], opp_money=o0.farms[1 - seat]["money"],
                    market_wheat=o0.market["inventory"]["WHEAT"])
    TAP["fl"] = None
    shops = list(env.obs(0)["town"]["unlocked_shops"])
    ch = m._IMPL.chassis
    base_router = ch.router
    routes = sorted(ch.routes)

    def branch(r):
        if r is not None:
            ch.router = lambda obs, step, st, r=r: (base_router(obs, step, st), r)[1] if DECISION <= step < 648 else base_router(obs, step, st)
        chosen = None
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
            if chosen is None:
                chosen = ch.players.get(seat, {}).get("route")
            env.step(*((a, b) if seat == 0 else (b, a)))
        return chosen, env.money[seat], env.money[1 - seat]

    res = fork_branches([None] + routes, branch, max_parallel=1)
    default = res[0]
    return {"job": j, "kind": kind, "opponent": opp_name, "seed": seed, "seat": seat, "shops2": shops[:2], "shops": shops,
            "d_codes": [int(x) for x in stream.d], "s_codes": [int(x) for x in stream.s],
            "default_route": None if isinstance(default, Exception) else default[0],
            "results": {str(r): [v[1], v[2]] for r, v in zip(routes, res[1:]) if not isinstance(v, Exception)}}


def main():
    out, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    if len(sys.argv) > 5:
        TAPES.extend(load_tapes(sys.argv[5]))
        print("tapes", len(TAPES), flush=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["job"] for l in open(out) if l.strip()}
    jobs = [j for j in range(a, b) if j not in done]
    import time
    deadline = time.time() + float(os.environ.get("ROUTE_MAX_HOURS", "1e9")) * 3600
    with warm_pool(procs, maxtasksperchild=1) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run_job, jobs):
            if time.time() > deadline:
                print("time budget reached", flush=True)
                break
            f.write(json.dumps(r) + "\n")
            f.flush()
            best = max(r["results"].items(), key=lambda kv: kv[1][0] - kv[1][1])
            print(json.dumps({"job": r["job"], "opp": r["opponent"], "default": r["default_route"],
                              "best_by_diff": best[0], "n": len(r["results"])}), flush=True)


if __name__ == "__main__":
    main()
