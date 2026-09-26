"""A track promotion re-check (docs/PLAN_RL.md 5): >= 300 fresh paired games, disjoint from the training-time EVAL
suite: held-out tapes of the whole-DB pools (rl/opp_mix.export_v2 "eval" split) + live / self games on new seeds.

python -m tools.v5_recheck <policy.pt> <pools_v2.pkl> <vocab.npz> <out.json> [rules=0.25] [procs=4]
rules: comma list of "margin" or "margin/c" (c = LCB coefficient of the ensemble, rl/v5_policy.PolicyRunner); the
ensemble size is read from the weights. For every rule: win rate / money difference vs v4b on the same games (paired SE), per category, head-to-head.
"""
import json
import sys

import numpy as np

import rl.v5_rl as V
from infra.fork import warm_pool
from model.opp_data import Vocab
from rl.opp_mix import OppMix, load_pools

COUNTS = {"near": 120, "loss": 23, "top": 40, "live": 70, "self": 60}


def main():
    w, pools, vocab, out = sys.argv[1:5]
    rules = [tuple(float(y) for y in (x.split("/") + ["0"])[:2]) for x in (sys.argv[5] if len(sys.argv) > 5 else "0.25").split(",")]
    import torch
    V.G["ens"] = torch.load(w, map_location="cpu")["head.weight"].shape[0] // 15
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    V.G["vocab"] = Vocab(vocab)
    ev = load_pools(pools, "eval")
    counts = {k: min(v, len(ev.get(k, []))) if k in ev else v for k, v in COUNTS.items()}
    suite = OppMix(ev, seed_base=1_700_000_000).eval_specs(counts, start=3 * 10 ** 6)
    with warm_pool(procs) as pool:
        base = {j: (kind, d) for _, j, kind, d in pool.imap_unordered(V.eval_task, [("base", j, s, None) for j, s in suite])}
    res = {"n": len(suite), "base_win": float(np.mean([1.0 if d > 0 else 0.5 if d == 0 else 0.0 for _, d in base.values()]))}
    for m, c in rules:
        V.G["margin"], V.G["lcb_c"], V.G["net_key"] = m, c, None
        with warm_pool(procs) as pool:
            rec = V.evaluate(pool, suite, base, w, 40, 0.0)
        rec["margin"], rec["c"] = m, c
        res[f"{m}/{c}"] = rec
        print(json.dumps(rec), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
