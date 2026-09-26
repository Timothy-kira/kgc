"""Per-opponent-program route values (H5 + H6: route choice conditioned on the identified opponent program).

python -m tools.program_routes <near_programs.json> <out.jsonl> [top_k=6] [per_program=30] [procs=4] [db_dir]
The ladder band is dominated by a few deterministic programs (top-15 two-day market signatures cover 56% of
2100-2700 player-games). For each of the top_k programs, take its recorded episodes, replay the program's tape
open-loop in its seat (the market is the only channel) on the episode's seed, play v4b's chassis (cha22 +
clamp_sells) in the other seat to step 144, then fork over all routes to the end. Output as tools/route_data.
"""
import json
import sys

import tools.route_data as RD
from agent.loader import call_adapter, entry_name, load_module
from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool

JOBS = []
SETTINGS = {"clamp_sells": True}


def run(i):
    best_effort()
    prog, eid, seed, cfg, acts, k = JOBS[i]
    seat = 1 - k
    m = load_module(RD.CHA22)
    m._IMPL.chassis.cfg.update(SETTINGS)
    me = call_adapter(getattr(m, entry_name(m)))
    opp = RD.Tape(acts, k)
    env = FarmEnv(seed, cfg)
    while env.step_count < RD.DECISION and not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    ch = m._IMPL.chassis
    base = ch.router
    routes = sorted(ch.routes)
    shops = list(env.obs(0)["town"]["unlocked_shops"])

    def branch(r):
        if r is not None:
            ch.router = lambda obs, step, st, r=r: (base(obs, step, st), r)[1] if RD.DECISION <= step < 648 else base(obs, step, st)
        chosen = None
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
            if chosen is None:
                chosen = ch.players.get(seat, {}).get("route")
            env.step(*((a, b) if seat == 0 else (b, a)))
        return chosen, env.money[seat], env.money[1 - seat]

    res = fork_branches([None] + routes, branch, max_parallel=1)
    d = res[0]
    return {"program": prog, "episode": eid, "seed": seed, "seat": seat, "shops2": shops[:2],
            "default_route": None if isinstance(d, Exception) else d[0],
            "results": {str(r): [v[1], v[2]] for r, v in zip(routes, res[1:]) if not isinstance(v, Exception)}}


def main():
    progs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    from data.replay_db import ReplayDB, _unz
    db = ReplayDB(sys.argv[6])
    for pi, p in enumerate(progs[:top_k]):
        for eid, k in p["episodes"][:per]:
            row = db.row(eid)
            cfg = {kk: v for kk, v in json.loads(row["config"]).items() if v is not None}
            JOBS.append((pi, eid, int(row["seed"]), cfg, _unz(row["actions_zstd"]), k))
    print("jobs", len(JOBS), flush=True)
    with warm_pool(procs, maxtasksperchild=1) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run, range(len(JOBS))):
            f.write(json.dumps(r) + "\n")
            f.flush()
            res = r["results"]
            dflt = res.get(str(r["default_route"]))
            print(json.dumps({"program": r["program"], "episode": r["episode"], "default": r["default_route"],
                              "default_diff": None if dflt is None else round(dflt[0] - dflt[1])}), flush=True)


if __name__ == "__main__":
    main()
