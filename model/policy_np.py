"""Numpy twin of model/net.py for step-by-step inference inside the Kaggle agent.

Incremental: keeps the last WIN step embeddings (sliding window KV) and the
fast weight W_fast, which is updated at every day boundary from the tracker's
day targets exactly like the training-time unrolled update.
"""
import math

import numpy as np

from agent.features import BASE_BIN, N_BINS

D, HEADS, WIN, N_PROD = 64, 4, 24, 9


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    from math import sqrt
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


try:
    from scipy.special import erf as _erf  # noqa
except Exception:  # pragma: no cover - numpy-only fallback (max abs err ~1e-7)
    def _erf(x):
        s = np.sign(x)
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
        return s * y


def _softmax(a, axis=-1):
    a = a - a.max(axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis, keepdims=True)


class NumpyPolicy:
    def __init__(self, weights, temperature=1.0, rng=None):
        w = dict(np.load(weights)) if isinstance(weights, str) else weights
        self.w = {k: v.astype(np.float64) for k, v in w.items()}
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.win = []                      # recent pre-SWA embeddings z
        self.W = self.w["w_fast0"].copy()
        self.day_u = []                    # u vectors of the current day
        self.cur_day = 0
        self.eta = 0.5 / (1.0 + math.exp(-float(self.w["log_eta"])))

    # ---- building blocks
    def _lin(self, x, name):
        y = x @ self.w[name + ".weight"].T
        b = self.w.get(name + ".bias")
        return y + b if b is not None else y

    def _block(self, pre, x, q_rows=None):
        """Pre-LN transformer block; if q_rows is given only those rows are queries (causal window)."""
        w = self.w
        xn = _ln(x, w[pre + "ln1.weight"], w[pre + "ln1.bias"])
        qkv = self._lin(xn, pre + "qkv")
        T = x.shape[0]
        dh = D // HEADS
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        rows = np.arange(T) if q_rows is None else np.asarray(q_rows)
        out = np.zeros((len(rows), D))
        for hh in range(HEADS):
            sl = slice(hh * dh, (hh + 1) * dh)
            a = q[rows, sl] @ k[:, sl].T / math.sqrt(dh)
            out[:, sl] = _softmax(a) @ v[:, sl]
        y = x[rows] + self._lin(out, pre + "proj")
        yn = _ln(y, w[pre + "ln2.weight"], w[pre + "ln2.bias"])
        return y + self._lin(_gelu(self._lin(yn, pre + "fc1")), pre + "fc2")

    # ---- forward for one step
    def forward(self, P, G, tracker=None, step=None):
        w = self.w
        P = np.asarray(P, np.float64)
        G = np.asarray(G, np.float64)
        # TTT fast-weight update at day boundaries (targets for finished days come from the tracker)
        self.last_targets = []
        if tracker is not None:
            for d, tgt in tracker.pop_day_targets():
                self._ttt_update(tgt)
                self.last_targets.append((d, tgt))
        x = np.concatenate([self._lin(P, "pin") + w["pemb"], self._lin(G[None], "gin")], 0)
        for i in range(2):
            x = self._block(f"enc.{i}.", x)
        po, z = x[:N_PROD], x[N_PROD]
        self.win.append(z)
        if len(self.win) > WIN:
            self.win.pop(0)
        h = self._block("swa.", np.array(self.win), q_rows=[len(self.win) - 1])[0]
        u = _gelu(self._lin(_ln(h, w["ln_t.weight"], w["ln_t.bias"]), "w1"))
        self.day_u.append(u)
        hp = h + self.W @ u
        hx = np.repeat(hp[None], N_PROD, 0)
        feat = np.concatenate([po, hx], -1)
        if "thead.0.weight" in w:
            tl = self._lin(_gelu(self._lin(feat, "thead.0")), "thead.2")
            feat = np.concatenate([feat, _softmax(tl)], -1)
        hid = _gelu(self._lin(feat, "head.0"))
        logits = self._lin(hid, "head.2")
        v = self._lin(_gelu(self._lin(hp, "vhead.0")), "vhead.2")
        return logits, v

    def _ttt_update(self, tgt):
        if not self.day_u:
            return
        ub = np.mean(self.day_u, 0)
        Py = self.w["p_y.weight"]
        err = Py @ (self.W @ ub) - tgt
        self.W = self.W - self.eta * np.outer(Py.T @ err, ub)
        self.day_u = []

    # ---- policies usable by agent.controller.MarketController
    def _mask(self, P, base_bins):
        stock = P[:, 2] > 0
        m = np.ones((N_PROD, N_BINS), bool)
        m[~stock, :] = False
        m[~stock, 0] = True
        m[~stock, BASE_BIN] = True
        return m

    def act_sample(self, P, G, tracker, base_bins):
        logits, v = self.forward(P, G, tracker)
        m = self._mask(P, base_bins)
        self.last_mask = m
        lg = np.where(m, logits / self.temperature, -1e9)
        pr = _softmax(lg)
        bins = np.array([self.rng.choice(N_BINS, p=pr[k]) for k in range(N_PROD)])
        logp = np.log(pr[np.arange(N_PROD), bins] + 1e-12)
        return bins, (logp, v, m, self.last_targets)

    def act_greedy(self, P, G, tracker, base_bins):
        logits, v = self.forward(P, G, tracker)
        m = self._mask(P, base_bins)
        bins = np.where(m, logits, -1e9).argmax(-1)
        return bins, (np.zeros(N_PROD), v, m, self.last_targets)
