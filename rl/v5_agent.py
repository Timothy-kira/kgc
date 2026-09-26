"""v5 agent: v4b base (cha22 + clamp_sells) with a learned residual on sell decisions (docs/PLAN_RL.md).

Every 6 hours the policy picks, for each controlled product, 0 = follow cha22, 1 = hold (drop that product's SELL
orders for the next 6 steps), 2 = dump (sell half of the shed stock now, on top of cha22's orders). Guards:
money < 3000 -> no hold (cha22's tape funds its purchases from its sales; selling more is always allowed);
shed + carried > 85 -> no hold; fertilizer keeps a reserve of 10. The headroom probe (tools/v5_headroom.py)
found whole-stock dumps at -1.5k on average with 25-50% positive cases, hence half.

Features (FEAT_DIM per decision): global state, per controlled product (price, market inventory deviation, our
stock, cha22's planned sells over the next 6/24 steps, opponent's inferred flows over 6/24/72 steps, town drain,
active hold) and a short context for the other products. The opponent's S/D event streams feed the policy's Engram
layer (hash rows computed here with the official hasher).
"""
import collections
import copy
import importlib
import math

import numpy as np

from agent.loader import call_adapter, entry_name, load_module
from agent.opp_events import OppEventStream, drain, infer_flows
from model.engram import EngramLayout, NgramHasher

CTRL = ("MILK", "WOOL", "MELON", "STRAWBERRY", "FERTILIZER")
CTX = ("WHEAT", "CARROT", "TOMATO", "EGG")
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160,
        "WOOL": 200, "FERTILIZER": 100}
T_SCALE = {"WHEAT": 400, "CARROT": 450, "TOMATO": 200, "STRAWBERRY": 100, "MELON": 300, "EGG": 332, "MILK": 122,
           "WOOL": 105, "FERTILIZER": 200}
N_ACT = 3
FOLLOW, HOLD, DUMP = 0, 1, 2
DECIDE_EVERY = 6
WINDOW = 8
FEAT_DIM = 8 + 10 * len(CTRL) + 3 * len(CTX)
BASE_SETTINGS = {"clamp_sells": True}
MIN_MONEY = 3000
SHED_HOLD_MAX = 85
FERT_RESERVE = 10
BUCKET_START = 2 ** 13

try:
    K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
except Exception:                                   # the Kaggle runtime ships it; keep importable elsewhere
    K = None


def engram_layout(vocab_sizes):
    return EngramLayout.build((1, 3), 4, 4, 32, BUCKET_START, vocab_sizes)


class V5Agent:
    """Callable Kaggle agent. `policy_fn(inputs) -> (actions[5], logp)` decides at decision steps; `override`
    (a length-5 action list) replaces the policy for the next decision only (used by fork branches)."""

    def __init__(self, vocab, policy_fn=None, cha22_path="league/cha22.py", settings=None):
        self.m = load_module(cha22_path)
        self.m._IMPL.chassis.cfg.update(BASE_SETTINGS if settings is None else settings)
        self.base = call_adapter(getattr(self.m, entry_name(self.m)))
        self.chassis = self.m._IMPL.chassis
        self.vocab = vocab
        self.hasher = NgramHasher(engram_layout(vocab.sizes), pad_id=0)
        self.policy_fn = policy_fn
        self.stream = OppEventStream()
        self.flows = {p: collections.deque(maxlen=72) for p in CTRL}
        self.prev_obs = self.prev_action = None
        self.synced = -1
        self.hist = collections.deque(maxlen=WINDOW)
        self.hold_until = {p: -1 for p in CTRL}
        self.override = None
        self.log = []                                      # (step, inputs, actions, logp) of every decision

    # ------------------------------------------------------------------ state tracking
    def sync(self, obs):
        step = int(obs["step"])
        if step == self.synced:
            return
        self.synced = step
        if self.prev_obs is not None and K is not None:
            me = int(obs["player"])
            try:
                fl = infer_flows(K, self.prev_obs, self.prev_action, obs["market"]["inventory"])
            except Exception:
                fl = {}
            o = obs["farms"][1 - me]
            self.stream.push(int(self.prev_obs["step"]), fl, o, opp_money=o["money"],
                             market_wheat=obs["market"]["inventory"]["WHEAT"])
            for p in CTRL:
                self.flows[p].append(fl.get(p, 0))

    def _planned(self, p, step, h):
        st = self.chassis.players.get(int(self._me), {})
        route = st.get("route")
        if route is None or route not in self.chassis.routes:
            return 0
        return self.chassis.future_sells(route, p, step) - self.chassis.future_sells(route, p, step + h)

    def features(self, obs):
        me = int(obs["player"])
        self._me = me
        step, hour = int(obs["step"]), int(obs["step"]) % 24
        farm, opp = obs["farms"][me], obs["farms"][1 - me]
        inv, shed = obs["market"]["inventory"], obs["private"]["shed"]
        carried = sum(sum(i.values()) for i in obs["private"].get("inventories") or [])
        shed_tot = sum(shed.values())
        dr = collections.Counter()
        for s in range(24):
            dr.update(drain(step + s, obs["town"]["unlocked_shops"]))
        f = [step / 720, hour / 24, math.log1p(max(0, farm["money"])) / 12, math.log1p(max(0, opp["money"])) / 12,
             max(-3.0, min(3.0, (farm["money"] - opp["money"]) / 50000)), shed_tot / 100, carried / 50,
             float(farm["money"] < MIN_MONEY)]
        for p in CTRL:
            price = obs["market"].get("prices", {}).get(p) or BASE[p]
            fl = list(self.flows[p])
            f += [price / BASE[p], max(-3.0, min(3.0, (inv[p] - 10000) / T_SCALE[p])), shed.get(p, 0) / 50,
                  self._planned(p, step, 6) / 20, self._planned(p, step, 24) / 50,
                  sum(fl[-6:]) / 10, sum(fl[-24:]) / 30, sum(fl) / 60, dr[p] / 12, float(self.hold_until[p] > step)]
        for p in CTX:
            price = obs["market"].get("prices", {}).get(p) or BASE[p]
            f += [price / BASE[p], max(-3.0, min(3.0, (inv[p] - 10000) / T_SCALE[p])), shed.get(p, 0) / 50]
        return np.array(f, np.float32)

    def engram_rows(self):
        s = self.vocab.s_ids(self.stream.s[-4:]) if self.stream.s else np.array([-1])
        d = self.vocab.d_ids(self.stream.d[-4:]) if self.stream.d else np.array([-1])
        return self.hasher(np.asarray(s, np.int64), 0)[-1], self.hasher(np.asarray(d, np.int64), 1)[-1]

    def inputs(self, obs):
        """Policy inputs at a decision step (window of the last WINDOW decision tokens incl. this one)."""
        self.sync(obs)
        step = int(obs["step"])
        if not self.hist or self.hist[-1][0] != step:
            rs, rd = self.engram_rows()
            self.hist.append((step, self.features(obs), rs, rd))
        W = len(self.hist)
        feats = np.zeros((WINDOW, FEAT_DIM), np.float32)
        rows_s = np.zeros((WINDOW, self.hasher.layout.n_hash_cols), np.int64)
        rows_d = np.zeros_like(rows_s)
        mask = np.zeros(WINDOW, bool)
        for i, (_, f, rs, rd) in enumerate(self.hist):
            j = WINDOW - W + i
            feats[j], rows_s[j], rows_d[j], mask[j] = f, rs, rd, True
        return {"feats": feats, "rows_s": rows_s, "rows_d": rows_d, "mask": mask, "allowed": self.allowed(obs)}

    def allowed(self, obs):
        """[5, 3] bool: which actions each head may take (guards)."""
        me = int(obs["player"])
        a = np.ones((len(CTRL), N_ACT), bool)
        shed = obs["private"]["shed"]
        carried = sum(sum(i.values()) for i in obs["private"].get("inventories") or [])
        if obs["farms"][me]["money"] < MIN_MONEY or sum(shed.values()) + carried > SHED_HOLD_MAX:
            a[:, HOLD] = False
        for k, p in enumerate(CTRL):
            if self._dump_qty(p, shed) <= 0:
                a[k, DUMP] = False
        return a

    @staticmethod
    def _dump_qty(p, shed):
        return (shed.get(p, 0) - (FERT_RESERVE if p == "FERTILIZER" else 0)) // 2

    # ------------------------------------------------------------------ acting
    def decide(self, obs):
        inp = self.inputs(obs)
        if self.override is not None:
            acts, logp = list(self.override), None
            self.override = None
        elif self.policy_fn is not None:
            acts, logp = self.policy_fn(inp)
        else:
            acts, logp = [FOLLOW] * len(CTRL), 0.0
        acts = [a if inp["allowed"][k, a] else FOLLOW for k, a in enumerate(acts)]
        self.log.append((int(obs["step"]), inp, acts, logp))
        return acts

    def apply(self, obs, action, acts):
        step = int(obs["step"])
        shed = obs["private"]["shed"]
        dump = set()
        if acts is not None:
            for k, p in enumerate(CTRL):
                if acts[k] == HOLD:
                    self.hold_until[p] = step + DECIDE_EVERY
                elif acts[k] == DUMP:
                    self.hold_until[p] = -1
                    dump.add(p)
                else:
                    self.hold_until[p] = -1
        carried = sum(sum(i.values()) for i in obs["private"].get("inventories") or [])
        if sum(shed.values()) + carried > SHED_HOLD_MAX + 5:          # shed filling up: release holds
            self.hold_until = {p: -1 for p in CTRL}
        out, rest = [], []
        for p in CTRL:                                   # dump orders first (the engine keeps 10 per turn)
            if p in dump:
                q = self._dump_qty(p, shed)
                if q > 0:
                    out.append(["SELL", p, int(q)])
        for o in action.get("market") or []:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in CTRL and self.hold_until[o[1]] > step:
                continue
            rest.append(o)
        if not out and len(rest) == len(action.get("market") or []):
            return action                                # nothing changed: exactly the base action
        out += rest
        action = dict(action)
        action["market"] = out
        return action

    def __call__(self, obs, cfg=None):
        self.sync(obs)
        action = self.base(obs, cfg)
        try:
            acts = self.decide(obs) if int(obs["step"]) % DECIDE_EVERY == 0 else None
            action = self.apply(obs, action, acts)
        except Exception:
            pass
        self.prev_obs, self.prev_action = copy.deepcopy(obs), copy.deepcopy(action)
        return action
