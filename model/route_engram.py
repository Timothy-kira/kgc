"""H5: an Engram memory that replaces cha22's hand-written route tables (docs/PLAN_v4.1.md).

cha22 picks its route at step 144 from tables keyed by the first two shops (`_R108_SHOP_ROUTES`, `_R110_OLD_SHOPS`,
`_V92_TABLE`). Here the key is widened to n-grams over [opponent fingerprint, opponent day codes 0..5, shop pair]
and the value is learned: the official Engram lookup (model/engram.py) gated by a shop-pair context vector, then a
head that predicts, for every route, the final money difference relative to cha22's own choice (in units of 10k).
The head starts at zero, so before training the decision equals cha22's; a route is switched only when its
predicted gain exceeds `margin`.
"""
import numpy as np
import torch
from torch import nn

from model.engram import Engram, EngramLayout, NgramHasher

SHOP_NAMES = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP",
              "YARN_STORE"]
N_DAYS = 6
DEAD = -1


def shop_pair_id(shops2):
    ids = sorted(SHOP_NAMES.index(s) for s in shops2[:2] if s in SHOP_NAMES)
    ids = (ids + [8, 8])[:2]
    return ids[0] * 9 + ids[1]


class RouteVocab:
    """Compressed ids for opponent D codes (fingerprint + day codes); shop pairs get their own id range."""

    def __init__(self, codes=None, path=None, min_count=1):
        if path is not None:
            with np.load(path) as z:
                self.keys = z["keys"]
        else:
            import collections
            c = collections.Counter(codes)
            self.keys = np.array(sorted(k for k, n in c.items() if n >= min_count), np.int64)
        self.n_shop = 81
        self.size = 2 + self.n_shop + len(self.keys)

    def save(self, path):
        np.savez_compressed(path, keys=self.keys)

    def code_id(self, codes):
        codes = np.asarray(codes, np.int64)
        if not len(self.keys):
            return np.ones(len(codes), np.int64)
        i = np.searchsorted(self.keys, codes).clip(0, len(self.keys) - 1)
        return np.where(self.keys[i] == codes, 2 + self.n_shop + i, 1)

    def sequence(self, d_codes, shops2):
        """[fingerprint, day0..day5 (DEAD if missing), shop pair] as compressed ids."""
        d = list(d_codes[:1 + N_DAYS])
        ids = list(self.code_id(d)) + [DEAD] * (1 + N_DAYS - len(d))
        return np.array(ids + [2 + shop_pair_id(shops2)], np.int64)


class RouteEngram(nn.Module):
    def __init__(self, vocab_size, routes, dim=64, n_heads=4, head_dim=32, bucket_start=2 ** 13):
        super().__init__()
        self.routes = list(routes)
        self.layout = EngramLayout.build((1,), 4, n_heads, head_dim, bucket_start, (vocab_size,))
        self.hasher = NgramHasher(self.layout, pad_id=0)
        self.shop = nn.Embedding(82, dim)
        self.fp = nn.Embedding(vocab_size + 1, dim)
        self.engram = Engram(dim, 1, self.layout.rows(0), self.layout.n_hash_cols, head_dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, len(self.routes)))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def hash_rows(self, seqs):
        """seqs int64 [B, 8] -> rows [B, 2, cols]: n-grams ending at the shop token of the full sequence and of
        [fingerprint, shop] (so the pair (opponent program, shops) has its own key)."""
        full = self.hasher(seqs, 0)[:, -1]
        short = self.hasher(np.stack([seqs[:, 0], seqs[:, -1]], 1), 0)[:, -1]
        return np.stack([full, short], 1)

    def forward(self, seqs):
        seqs_t = torch.as_tensor(seqs)
        rows = torch.as_tensor(self.hash_rows(np.asarray(seqs)))
        ctx = self.shop(seqs_t[:, -1] - 2) + self.fp(seqs_t[:, 0].clamp_min(0))
        h = ctx[:, None, None, :].expand(-1, 2, 1, -1)                         # [B, 2 lookups, hc=1, dim]
        h = self.engram(h.reshape(-1, 1, 1, ctx.shape[-1]), rows.reshape(-1, 1, rows.shape[-1])[:, None, 0])
        h = h.reshape(len(seqs_t), 2, -1).mean(1)
        return self.head(h)                                                    # predicted gain vs cha22's route

    def choose(self, seq, default_route, margin=0.1):
        with torch.no_grad():
            pred = self.forward(seq[None])[0].numpy()
        if default_route not in self.routes:
            return default_route, 0.0
        d = pred[self.routes.index(default_route)]
        k = int(pred.argmax())
        gain = float(pred[k] - d)
        return (self.routes[k], gain) if gain > margin else (default_route, gain)
