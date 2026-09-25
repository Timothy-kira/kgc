"""Unit tests for model/engram.py against the official DeepSeek-V4.1-Flash algorithm.

python -m tools.test_engram   -> prints ENGRAM_TESTS_OK
"""
import numpy as np
import torch

from model.engram import (Engram, EngramLayout, NgramHasher, compute_hash_multipliers, dequantize_table,
                          quantize_table)


def official_hash(ids, mult, primes, offsets, n, pad, dead=-1):
    """Direct transcription of NgramHashState.forward for one layer, one sequence (torch, as in Flash)."""
    ids = torch.tensor(ids)
    L = len(ids)
    pos = torch.arange(L)
    toks, blocked = [], torch.zeros(L, dtype=torch.bool)
    for s in range(n):
        src = ids[(pos - s).clamp_min(0)]
        blocked = blocked | (pos < s) | (src == dead)
        toks.append(torch.where(blocked, torch.tensor(pad), src))
    prod = torch.stack(toks, -1) * torch.tensor(mult)
    rolling, out = prod[..., 0], []
    for i in range(1, n):
        rolling = torch.bitwise_xor(rolling, prod[..., i])
        out.append(rolling.unsqueeze(-1) % torch.tensor(primes[i - 1]))
    return (torch.cat(out, -1) + torch.tensor(offsets)).numpy()


def main():
    lay = EngramLayout.build((1, 3), max_ngram_size=4, n_heads=4, head_dim=32, bucket_start=2 ** 10,
                             vocab_sizes=(500, 300))
    flat = [p for layer in lay.primes for per in layer for p in per]
    assert len(set(flat)) == len(flat), "primes reused"
    h = NgramHasher(lay, pad_id=0)
    rng = np.random.default_rng(0)
    for i, v in enumerate(lay.vocab_sizes):
        ids = rng.integers(0, v, size=40)
        ids[17] = -1                                                   # a DEAD id breaks n-grams
        got = h(ids, i)
        ref = official_hash(ids, compute_hash_multipliers((lay.layer_ids[i],), 4, v)[0], lay.primes[i],
                            h.offsets[i], 4, 0)
        assert (got == ref).all(), "hash mismatch vs official"
        assert got.max() < lay.rows(i) and got.min() >= 0
        # disjoint ranges: column c always falls inside its own [offset, offset + prime)
        sizes = [p for per in lay.primes[i] for p in per]
        for c in range(lay.n_hash_cols):
            assert (got[:, c] >= h.offsets[i][c]).all() and (got[:, c] < h.offsets[i][c] + sizes[c]).all()
        # sequences shorter than the n-gram order (look-back falls off the start)
        short = ids[:2]
        ref_s = official_hash(short, compute_hash_multipliers((lay.layer_ids[i],), 4, v)[0], lay.primes[i],
                              h.offsets[i], 4, 0)
        assert (h(short, i) == ref_s).all(), "short-sequence hash mismatch"
        # batched call == per-sequence call; deterministic
        assert (h(np.stack([ids, ids]), i)[1] == got).all()
    # module: zero value -> identity; token_mask closes the gate
    m = Engram(dim=16, hc_mult=2, rows=lay.rows(0), n_hash_cols=lay.n_hash_cols, head_dim=32)
    x = torch.randn(2, 5, 2, 16)
    hid = torch.tensor(h(rng.integers(0, 500, size=(2, 5)), 0))
    assert torch.equal(m(x, hid), x), "fresh Engram must be the identity"
    torch.nn.init.normal_(m.wkv.weight)
    y = m(x, hid)
    assert not torch.allclose(y, x)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    assert torch.allclose(m(x, hid, mask), x), "closed gate must pass through"
    q, s = quantize_table(m.embed.weight.data)
    assert (dequantize_table(q, s) - m.embed.weight.data).abs().max() < 0.02
    print("ENGRAM_TESTS_OK")


if __name__ == "__main__":
    main()
