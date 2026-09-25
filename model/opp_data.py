"""Join opponent event data (data/opp_flow_extract.py) onto CLM trajectories for Engram training.

Per trajectory (before cropping, so n-grams see the full history) this attaches:
  eng_s  int64 [T, n_hash_cols]  S-stream hash rows (layer 1): n-gram of the opponent's events of transitions < t
  eng_d  int64 [T, n_hash_cols]  D-stream hash rows (layer 3): n-gram of the opponent's finished days / fingerprint
  opp_tgt int64 [T, H, 9]        CLM targets (agent/opp_events.opp_targets)
  opp_stock float32 [T, 9]       log1p of the opponent's shed before step t
Compressed vocab (official build_compressed_token_map analogue): raw codes seen >= min_count times get ids
2.., UNK = 1, pad = 0.
"""
import collections
import glob
import os

import numpy as np

from agent.opp_events import opp_targets
from model.engram import EngramLayout, NgramHasher

DEAD = -1
LAYER_S, LAYER_D = 1, 3


def build_vocab(files, out, min_count=3, max_size=(400000, 200000)):
    cs, cd = collections.Counter(), collections.Counter()
    for f in files:
        with np.load(f) as z:
            cs.update(z["s_code"].tolist())
            cd.update(z["d_code"].tolist())
    keep_s = sorted(c for c, n in cs.most_common(max_size[0]) if n >= min_count)
    keep_d = sorted(c for c, n in cd.most_common(max_size[1]) if n >= min_count)
    np.savez_compressed(out, s=np.array(keep_s, np.int64), d=np.array(keep_d, np.int64))
    return len(keep_s) + 2, len(keep_d) + 2


class Vocab:
    def __init__(self, path):
        with np.load(path) as z:
            self.s, self.d = z["s"], z["d"]
        self.sizes = (len(self.s) + 2, len(self.d) + 2)

    @staticmethod
    def _map(keys, codes):
        codes = np.asarray(codes, np.int64)
        i = np.searchsorted(keys, codes).clip(0, max(len(keys) - 1, 0))
        hit = (keys[i] == codes) if len(keys) else np.zeros(len(codes), bool)
        return np.where(hit, i + 2, 1)

    def s_ids(self, codes):
        return self._map(self.s, codes)

    def d_ids(self, codes):
        return self._map(self.d, codes)


def layout_for(vocab, n_heads=4, head_dim=32, bucket_start=2 ** 15, max_ngram=4):
    return EngramLayout.build((LAYER_S, LAYER_D), max_ngram, n_heads, head_dim, bucket_start, vocab.sizes)


def d_availability(n_d, T):
    """Step from which each D token is visible: fingerprint after transition 2, day d after transition 24d+23."""
    avail = [3] + [24 * (k + 1) for k in range(n_d - 1)]
    return np.array(avail[:n_d], np.int64)


def step_rows(hasher, vocab, s_codes, d_codes, T):
    """Hash rows per decision step t in [0, T)."""
    s = vocab.s_ids(s_codes)[:T]
    seq_s = np.concatenate([[DEAD], s])[:T]                     # step t sees events of transitions < t
    rows_s = hasher(seq_s, 0)
    d = vocab.d_ids(d_codes)
    rows_d_all = hasher(np.concatenate([[DEAD], d]), 1)         # index 0 = nothing visible yet
    n_vis = np.searchsorted(d_availability(len(d), T), np.arange(T), side="right")
    return rows_s, rows_d_all[n_vis]


class OppIndex:
    """(episode_id, seat) -> location in the opponent-event npz files, with a small file cache."""

    def __init__(self, pattern, vocab_path, cache=4):
        self.files = sorted(glob.glob(pattern))
        self.loc = {}
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                for k, (e, s) in enumerate(zip(z["episode_id"], z["seat"])):
                    self.loc[(int(e), int(s))] = (fi, k)
        self.vocab = Vocab(vocab_path)
        self.layout = layout_for(self.vocab)
        self.hasher = NgramHasher(self.layout, pad_id=0)
        self._cache = collections.OrderedDict()
        self.cache = cache

    def _file(self, fi):
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        with np.load(self.files[fi]) as z:
            d = {k: z[k] for k in z.files}
        self._cache[fi] = d
        if len(self._cache) > self.cache:
            self._cache.popitem(last=False)
        return d

    def attach(self, tr):
        key = (tr.get("episode_id", -1), tr.get("seat", -1))
        if key not in self.loc:
            return False
        fi, k = self.loc[key]
        d = self._file(fi)
        s0, s1 = d["t_off"][k], d["t_off"][k + 1]
        d0, d1 = d["d_off"][k], d["d_off"][k + 1]
        T = len(tr["prod"])
        flow = d["flow"][s0:s1].astype(np.int64)
        if len(flow) < T - 1:
            return False
        tr["eng_s"], tr["eng_d"] = step_rows(self.hasher, self.vocab, d["s_code"][s0:s1], d["d_code"][d0:d1], T)
        tgt = opp_targets(flow)
        tr["opp_tgt"] = np.concatenate([tgt, tgt[-1:].repeat(max(0, T - len(tgt)), 0)])[:T]
        shed = d["opp_shed"][s0:s1].astype(np.float32)
        before = np.concatenate([np.zeros((1, shed.shape[1]), np.float32), shed])[:T]
        tr["opp_stock"] = np.log1p(before)
        return True


OPP_KEYS = ("eng_s", "eng_d", "opp_tgt", "opp_stock")
