"""Round-trip test of action tokenisation on replay actions (+ grammar acceptance).

python -m tools.test_tokens <db_dir> [n_episodes]
"""
import collections
import sys

import numpy as np

from agent.action_tokens import Grammar, decode, encode
from data.replay_db import ReplayDB


def canon(a):
    """What the environment actually sees (framework passes dicts through as-is)."""
    return {"farmer": a.get("farmer"), "hands": list(a.get("hands") or []), "market": list(a.get("market") or [])}


def main():
    db = ReplayDB(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    stats = collections.Counter()
    lens = []
    for k, r in enumerate(db.iter_rows()):
        if k >= n:
            break
        for pair in r["actions"]:
            for a in pair:
                if not isinstance(a, dict):
                    stats["non_dict"] += 1
                    continue
                ids = encode(a)
                lens.append(len(ids))
                back = decode(ids)
                if back != canon(a):
                    stats["mismatch"] += 1
                    if stats["mismatch"] < 3:
                        print("MISMATCH", a, back)
                g = Grammar(max_hands=len(a.get("hands") or []))
                ok = True
                for t in ids:
                    if not g.mask()[t]:
                        ok = False
                        break
                    g.feed(t)
                stats["grammar_ok" if ok and g.done else "grammar_bad"] += 1
                stats["actions"] += 1
    print(dict(stats), "tokens/step mean", np.mean(lens), "p99", np.percentile(lens, 99), "max", max(lens))


if __name__ == "__main__":
    main()
