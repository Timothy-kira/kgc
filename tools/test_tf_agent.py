"""Live Engram rows (agent/tf_agent.rows_from_tail) == training rows (model/opp_data.step_rows)."""
import sys

import numpy as np

from agent.tf_agent import rows_from_tail
from model.engram import NgramHasher
from model.opp_data import Vocab, d_availability, layout_for, step_rows


def main(vocab_path):
    v = Vocab(vocab_path)
    h = NgramHasher(layout_for(v), pad_id=0)
    rng = np.random.default_rng(0)
    T = 300
    s_codes = rng.choice(v.s[:50], T)
    d_codes = rng.choice(v.d[:50], 13)
    rs, rd = step_rows(h, v, s_codes, d_codes, T)
    s_ids, d_ids = v.s_ids(s_codes), v.d_ids(d_codes)
    avail = d_availability(len(d_ids), T)
    for t in range(T):
        a = rows_from_tail(h, s_ids[:t], 0, h.layout.max_ngram_size)
        n_vis = int((avail <= t).sum())
        b = rows_from_tail(h, d_ids[:n_vis], 1, h.layout.max_ngram_size)
        assert (a == rs[t]).all(), t
        assert (b == rd[t]).all(), t
    print("TF_AGENT_ROWS_OK")


if __name__ == "__main__":
    main(sys.argv[1])
