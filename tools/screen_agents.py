"""Screen harvested agent files: loadability, per-step time and strength vs a reference agent.

python -m tools.screen_agents <agents_dir> <reference.py> [seeds=2] [procs=4] [out.json]
Each candidate plays both seats on each seed against the reference (Kaggle entry rule, agent/loader.py) on a
warm fork pool under SCHED_IDLE. A game is aborted after 240 s. Also computes a lineage fingerprint: Jaccard
similarity of top-level def/class names with the reference.
"""
import ast
import glob
import json
import os
import signal
import sys
import time

from agent.loader import load_agent
from env.fast_env import FarmEnv
from infra.fork import best_effort, warm_pool


def names(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return set()
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _alarm(*_):
    raise TimeoutError


def play(job):
    cand, ref, seed, seat = job
    best_effort()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(240)
    try:
        fc, fr = load_agent(cand), load_agent(ref)
        env = FarmEnv(seed)
        tc, n = 0.0, 0
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            t = time.time()
            a_c = fc(o[seat], env.config)
            tc += time.time() - t
            n += 1
            a_r = fr(o[1 - seat], env.config)
            env.step(*((a_c, a_r) if seat == 0 else (a_r, a_c)))
        signal.alarm(0)
        m = env.money
        return cand, seed, seat, m[seat] - m[1 - seat], m[seat], 1000 * tc / max(n, 1), None
    except BaseException as e:
        signal.alarm(0)
        return cand, seed, seat, None, None, None, repr(e)[:120]


def main():
    d, ref = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(d, "screen.json")
    cands = sorted(glob.glob(os.path.join(d, "*.py")))
    ref_names = names(ref)
    jobs = [(c, ref, 9100 + s, seat) for c in cands for s in range(seeds) for seat in (0, 1)]
    res = {}
    t0 = time.time()
    with warm_pool(procs) as pool:
        for c, seed, seat, diff, money, ms, err in pool.imap_unordered(play, jobs):
            r = res.setdefault(c, {"diffs": [], "money": [], "ms": [], "errors": []})
            if err:
                r["errors"].append(err)
            else:
                r["diffs"].append(diff); r["money"].append(money); r["ms"].append(ms)
    rows = []
    for c, r in res.items():
        nm = names(c)
        jac = len(nm & ref_names) / max(len(nm | ref_names), 1)
        n = len(r["diffs"])
        rows.append({"agent": os.path.basename(c), "games": n, "errors": r["errors"][:2],
                     "wins": sum(x > 0 for x in r["diffs"]), "mean_diff": round(sum(r["diffs"]) / n) if n else None,
                     "mean_money": round(sum(r["money"]) / n) if n else None,
                     "ms_step": round(max(r["ms"]), 2) if n else None, "jaccard_ref": round(jac, 3)})
    rows.sort(key=lambda x: -(x["mean_diff"] if x["mean_diff"] is not None else -1e9))
    json.dump(rows, open(out, "w"), indent=1)
    for x in rows:
        print(json.dumps(x))
    print(f"{len(jobs)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
