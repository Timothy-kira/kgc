"""Can a test-time lookahead pick cha22's route? (H5 without an ex-ante table)

python -m tools.lookahead_eval <route_data.jsonl> <out.jsonl> [n_games=40] [procs=3] [db_dir]
For games already in route-value data (true value of every route vs the real opponent), rebuild the job
(tools/route_data.job_spec, same env vars), play to step 144 with the real opponent, then fork over all routes
with the opponent replaced by a PROXY (a fresh cha22 taking over the opponent seat at step 144) and play to the
end. The route that is best under the proxy is looked up in the real values: realised gain vs cha22's own route.
"""
import json
import os
import sys

import tools.route_data as RD
from agent.loader import call_adapter, entry_name, load_agent, load_module
from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool

ROWS = {}


DB = {}


def spec_from_row(r):
    """Rebuild the opponent from the stored row (the DB grows, so job ids no longer map to the same tapes)."""
    name = r["opponent"]
    if ":" in name and not name.startswith("league"):
        from data.replay_db import _unz
        eid = int(name.split(":")[1])
        row = DB["db"].row(eid)
        acts = _unz(row["actions_zstd"])
        cfg = {k: v for k, v in json.loads(row["config"]).items() if v is not None}
        return r["kind"], name, r["seed"], r["seat"], cfg, RD.Tape(acts, 1 - r["seat"])
    return r["kind"], name, r["seed"], r["seat"], None, None


def job(j):
    best_effort()
    kind, opp_name, seed, seat, cfg, tape = spec_from_row(ROWS[j])
    m = load_module(RD.CHA22)
    me = call_adapter(getattr(m, entry_name(m)))
    opp = tape if tape is not None else load_agent(opp_name)
    env = FarmEnv(seed, cfg)
    while env.step_count < RD.DECISION and not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    ch = m._IMPL.chassis
    base = ch.router
    routes = sorted(ch.routes)

    def branch(r):
        ch.router = lambda obs, step, st, r=r: (base(obs, step, st), r)[1] if RD.DECISION <= step < 648 else base(obs, step, st)
        proxy = load_agent(RD.CHA22)
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            a, b = me(o[seat], env.config), proxy(o[1 - seat], env.config)
            env.step(*((a, b) if seat == 0 else (b, a)))
        return env.money[seat] - env.money[1 - seat]

    sim = fork_branches(routes, branch, max_parallel=1)
    return j, {str(r): v for r, v in zip(routes, sim) if not isinstance(v, Exception)}


def main():
    src, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    if len(sys.argv) > 5:
        from data.replay_db import ReplayDB
        DB["db"] = ReplayDB(sys.argv[5])
    rows = [json.loads(l) for l in open(src) if l.strip()]
    rows = [r for r in rows if str(r.get("default_route")) in r["results"]][:n]
    ROWS.update({r["job"]: r for r in rows})
    by_job = {r["job"]: r for r in rows}
    gains, oracle = [], []
    with warm_pool(procs, maxtasksperchild=1) as pool, open(out, "a") as f:
        for j, sim in pool.imap_unordered(job, list(by_job)):
            r = by_job[j]
            real = {k: v[0] - v[1] for k, v in r["results"].items()}
            base = real[str(r["default_route"])]
            pick = max(sim, key=sim.get)
            g = real.get(pick, base) - base
            gains.append(g)
            oracle.append(max(real.values()) - base)
            f.write(json.dumps({"job": j, "kind": r["kind"], "pick": pick, "default": r["default_route"], "gain": g,
                                "sim": sim}) + "\n")
            f.flush()
            print(json.dumps({"job": j, "kind": r["kind"], "gain": round(g), "mean_gain": round(sum(gains) / len(gains)),
                              "oracle_mean": round(sum(oracle) / len(oracle)), "n": len(gains)}), flush=True)


if __name__ == "__main__":
    main()
