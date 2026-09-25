"""Engram conditional memory, ported from DeepSeek-V4.1-Flash `inference/engram.py` + `model.py` (class Engram).

Official mechanism, kept as is:
  * ids are first mapped to a compressed vocabulary; each position is hashed together with the
    `max_ngram_size - 1` ids before it (look-back stops at the sequence start and at DEAD ids: pad instead);
  * one odd multiplier per (engram layer, look-back), from `default_rng(10007 * layer_id)`, bounded so that
    `id * multiplier` cannot overflow int64; a rolling XOR over the look-backs gives the 2..N-gram hashes;
  * every (layer, n-gram size, head) owns a distinct prime modulus (searched upward from `vocab_size`, never
    reused) and an offset, so all ranges are disjoint inside one table per layer;
  * lookup of (N-1)*heads rows -> `wkv` -> one key per hyper-connection copy + one shared value;
    gate = sigmoid(signed_sqrt(rms-normalised <h * q_w * k_w, key> / sqrt(dim))); out = h + gate * value;
    `token_mask` False closes the gate.

Kaggriculture adaptation (docs/PLAN_v4.1.md): the hashed "tokens" are compressed opponent events, not the
decoder's input ids (observation tokens carry their content in continuous features). Each game step gets one
hash row that is shared by all of that step's decoder positions. The value half of `wkv` starts at zero, so a
freshly inserted Engram leaves the pretrained network unchanged.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _isprime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n ** 0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def find_next_prime(start, seen):
    """The smallest prime above `start` that has not been handed out yet (official helper)."""
    c = start + 1
    while not _isprime(c) or c in seen:
        c += 1
    return c


def compute_hash_multipliers(layer_ids, max_ngram_size, vocab_size):
    """One odd multiplier per (layer, look-back) from a per-layer RNG (official)."""
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        rows.append(g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple
    primes: tuple            # [layer][n-gram size - 2][head]
    n_heads: int
    head_dim: int
    vocab_sizes: tuple       # compressed vocab size of the stream each layer hashes

    @classmethod
    def build(cls, layer_ids, max_ngram_size=4, n_heads=4, head_dim=32, bucket_start=2 ** 15, vocab_sizes=None):
        primes, seen = [], set()
        for _ in layer_ids:
            per = []
            for _ in range(max_ngram_size - 1):
                sizes, cur = [], bucket_start - 1
                for _ in range(n_heads):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(tuple(sizes))
            primes.append(tuple(per))
        return cls(max_ngram_size, tuple(layer_ids), tuple(primes), n_heads, head_dim, tuple(vocab_sizes))

    def rows(self, i):
        return int(sum(p for per in self.primes[i] for p in per))

    @property
    def n_hash_cols(self):
        return (self.max_ngram_size - 1) * self.n_heads


class NgramHasher:
    """Hash ids of the n-grams ending at each position of a compressed-id stream (numpy, official algorithm).

    `ids`: int64 [..., L] compressed ids of one stream (DEAD = -1 breaks n-grams); layer index `i` selects the
    multipliers / primes. Returns int64 [..., L, n_hash_cols] row indices into that layer's table."""
    DEAD = -1

    def __init__(self, layout, pad_id=0):
        self.layout = layout
        self.pad_id = pad_id
        self.mult = [compute_hash_multipliers((lid,), layout.max_ngram_size, v)[0]
                     for lid, v in zip(layout.layer_ids, layout.vocab_sizes)]
        self.primes = [np.array(layout.primes[i], np.int64) for i in range(len(layout.layer_ids))]
        self.offsets = [np.cumsum([0] + [p for per in layout.primes[i] for p in per][:-1]).astype(np.int64)
                        for i in range(len(layout.layer_ids))]

    def __call__(self, ids, i):
        ids = np.asarray(ids, np.int64)
        n = self.layout.max_ngram_size
        L = ids.shape[-1]
        toks, blocked = [], np.zeros(ids.shape, bool)
        pos = np.arange(L)
        for shift in range(n):
            src = np.concatenate([np.full(ids.shape[:-1] + (shift,), self.DEAD, np.int64), ids[..., :L - shift]], -1) \
                if shift else ids
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            toks.append(np.where(blocked, self.pad_id, src))
        prod = np.stack(toks, -1) * self.mult[i]                       # [..., L, n]
        rolling, out = prod[..., 0], []
        for k in range(1, n):
            rolling = np.bitwise_xor(rolling, prod[..., k])
            out.append(rolling[..., None] % self.primes[i][k - 1])      # [..., L, heads]
        return np.concatenate(out, -1) + self.offsets[i]


class Engram(nn.Module):
    """Official Engram module (per engram layer); x: [B, L, hc, dim]."""

    def __init__(self, dim, hc_mult, rows, n_hash_cols, head_dim, eps=1e-6):
        super().__init__()
        self.dim, self.hc_mult, self.eps, self.clamp_value = dim, hc_mult, eps, 1e-6
        self.embed = nn.Embedding(rows, head_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.wkv = nn.Linear(n_hash_cols * head_dim, dim * (hc_mult + 1), bias=False)
        with torch.no_grad():
            self.wkv.weight[hc_mult * dim:].zero_()                    # value = 0: pretrained net unchanged
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim))
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim))

    def forward(self, x, hash_ids, token_mask=None):
        """hash_ids: [B, L, n_hash_cols] int64; token_mask: [B, L] bool (False: pass through)."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def quantize_table(w, block=32):
    """int8 rows + per-block scales (the official tables are fp8 + block scales)."""
    w = w.float().unflatten(-1, (-1, block))
    scale = w.abs().amax(-1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.flatten(-2), scale.half()


def dequantize_table(q, scale, block=32):
    return (q.float().unflatten(-1, (-1, block)) * scale.float().unsqueeze(-1)).flatten(-2)
