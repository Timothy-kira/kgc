"""Can top teams' games serve as plans (tapes) executed by cha22's repair chassis?

python -m tools.tape_transfer <replay_db_dir> [teams=DSM,Mother-Goose] [seeds=6] [tapes_per_seed=2] [procs=4]
For each test seed: compute its first two shops (they depend on the seed only), pick top-team episodes recorded
in a world with the same first two shops, and play cha22 with its route replaced by that team's action
sequence: variant A from day 6 on (after cha22's own opening, i.e. where cha22's router itself switches),
variant B from step 0. Opponent: cha22. Reports money difference vs the unmodified cha22 on the same seed.
"""
import collections
import json
import sys

from agent.loader import call_adapter, entry_name, load_agent, load_module
from data.replay_db import ReplayDB, _unz
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool

TEAMS = {"DSM": 16732748, "Mother-Goose": 16730612, "MMPQ": 16681125, "Majkel1337": 16718819, "Boey": 16915014}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def shops_of(seed, day=6):
    env = FarmEnv(seed)
    while env.step_count < day * 24 + 1 and not env.done:
        env.step(PASS, PASS)
    return tuple((env.obs(0).get("town") or {}).get("unlocked_shops", [])[:2])


def cha22_with_tape(tape, from_step):
    m = load_module("league/cha22.py")
    ch = m._IMPL.chassis
    rid = 999
    ch.routes[rid] = [a if isinstance(a, dict) else PASS for a in tape]
    ch._future_sells = {}
    base_router = ch.router
    ch.router = (lambda obs, step, st: rid if step >= from_step else base_router(obs, step, st))
    return call_adapter(getattr(m, entry_name(m)))


def play(job):
    seed, tape, from_step, label = job
    best_effort()
    me = load_agent("league/cha22.py") if tape is None else cha22_with_tape(tape, from_step)
    opp = load_agent("league/cha22.py")
    env = FarmEnv(seed)
    while not env.done:
        env.step(me(env.obs(0), env.config), opp(env.obs(1), env.config))
    return seed, label, env.money[0] - env.money[1], env.money[0]


def main():
    db_dir = sys.argv[1]
    teams = (sys.argv[2] if len(sys.argv) > 2 else "DSM,Mother-Goose").split(",")
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    db = ReplayDB(db_dir)
    e = db.episodes()
    lib = collections.defaultdict(list)                   # shop key -> [(team, eid, seat)]
    for team in teams:
        tid = TEAMS[team]
        for k in (0, 1):
            sub = e[(e[f"team_id_{k}"] == tid)].sort_values("episode_id").tail(150)
            for r in sub.itertuples():
                won = (getattr(r, f"reward_{k}") or 0) > (getattr(r, f"reward_{1 - k}") or 0)
                if won:
                    lib[shops_of(int(r.seed))].append((team, int(r.episode_id), k))
    jobs = []
    s, used = 9900, 0
    while used < n_seeds and s < 9900 + 200:
        key = shops_of(s)
        if lib.get(key):
            jobs.append((s, None, 0, "cha22"))
            for team, eid, k in lib[key][:per]:
                acts = _unz(db.row(eid)["actions_zstd"])
                tape = [pair[k] for pair in acts]
                jobs.append((s, tape, 144, f"A:{team}:{eid}"))
                jobs.append((s, tape, 0, f"B:{team}:{eid}"))
            used += 1
        s += 1
    res = collections.defaultdict(dict)
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for seed, label, diff, money in pool.imap_unordered(play, jobs):
            res[seed][label] = (diff, money)
    agg = collections.defaultdict(list)
    for seed, r in sorted(res.items()):
        base = r["cha22"][1]
        for label, (diff, money) in sorted(r.items()):
            if label != "cha22":
                agg[label[0]].append(money - base)
                print(json.dumps({"seed": seed, "tape": label, "diff_vs_opp": round(diff), "money": round(money),
                                  "money_minus_cha22": round(money - base)}))
    for v, xs in agg.items():
        print(f"variant {v}: mean money vs unmodified cha22 {sum(xs) / len(xs):+.0f} over {len(xs)} games, better in {sum(x > 0 for x in xs)}")


if __name__ == "__main__":
    main()
