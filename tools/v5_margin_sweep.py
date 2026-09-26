"""A track: paired eval of a trained v5 advantage policy at several acting margins (rl/v5_rl.py --obj adv).

python -m tools.v5_margin_sweep <policy.pt> <pools.pkl> <vocab.npz> <base_eval.json> <margins comma> [procs=4]
Same held-out suite and v4b baseline as the training EVALs (60 paired jobs + 20 head-to-head games vs v4b).
"""
import json
import sys

from infra.fork import warm_pool
import rl.v5_rl as V
from model.opp_data import Vocab
from rl.opp_mix import OppMix, load_pools


def main():
    w, pools, vocab, base_p, margins = sys.argv[1:6]
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    V.G["vocab"] = Vocab(vocab)
    suite = OppMix(load_pools(pools, "eval"), seed_base=1_999_000_000).eval_specs()
    base = {int(k): tuple(v) for k, v in json.load(open(base_p)).items()}
    for m in [float(x) for x in margins.split(",")]:
        V.G["margin"], V.G["net_key"] = m, None
        with warm_pool(procs) as pool:
            rec = V.evaluate(pool, suite, base, w, 20, 0.0)
        rec["margin"] = m
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
