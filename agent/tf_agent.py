"""Pure-Transformer agent (B track, docs/PLAN_RL.md): CSA2 decoder + official Engram + CLM heads decide every slot;
no cha22 at inference.

Adds the live Engram inputs to agent/clm_agent.CLMAgent: after every transition the opponent's inventory flows are
inferred from our two observations and our own action (agent/opp_events.infer_flows, 99.87% exact), pushed into the
S / D event streams, and the step's hash rows are computed exactly as model/opp_data.step_rows does for training
(the step-t tokens see the S events of transitions < t and the D tokens visible by then; the rows of the last
max_ngram positions are all an n-gram of order <= max_ngram needs, see `rows_from_tail`).
"""
import copy
import importlib

import numpy as np
import torch

from agent.clm_agent import CLMAgent
from agent.opp_events import OppEventStream, infer_flows
from model.engram import NgramHasher
from model.opp_data import DEAD, LAYER_D, LAYER_S, layout_for

try:
    K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
except Exception:
    K = None


def rows_from_tail(hasher, ids, layer, n):
    """Hash row of the last position of [DEAD] + ids, using only the last n tokens (n = max n-gram order)."""
    seq = np.concatenate([[DEAD], np.asarray(ids, np.int64)])[-n:]
    return hasher(seq, layer)[-1]


class EngramLive:
    """Per-game opponent event state -> (rows_s, rows_d) for the current step."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.hasher = NgramHasher(layout_for(vocab), pad_id=0)
        self.n = self.hasher.layout.max_ngram_size
        self.reset()

    def reset(self):
        self.stream = OppEventStream()
        self.s_ids, self.d_ids = [], []
        self.flows = []

    def push(self, step, flows, opp_farm, opp_money, market_wheat):
        n_d = len(self.stream.d)
        self.stream.push(step, flows, opp_farm, opp_money=opp_money, market_wheat=market_wheat)
        self.s_ids.append(int(self.vocab.s_ids([self.stream.s[-1]])[0]))
        for c in self.stream.d[n_d:]:
            self.d_ids.append(int(self.vocab.d_ids([c])[0]))
        self.flows.append(flows)

    def rows(self):
        return rows_from_tail(self.hasher, self.s_ids, 0, self.n), rows_from_tail(self.hasher, self.d_ids, 1, self.n)


class TFAgent(CLMAgent):
    def __init__(self, net, vocab, temperature=0.0, **kw):
        self.live = EngramLive(vocab)
        super().__init__(net, temperature=temperature, **kw)

    def reset(self):
        super().reset()
        self.live.reset()
        self.prev_obs = self.prev_action = None
        self.cur_rows = None

    def _sync(self, obs):
        if self.prev_obs is not None and K is not None:
            me = int(obs["player"])
            try:
                fl = infer_flows(K, self.prev_obs, self.prev_action, obs["market"]["inventory"])
            except Exception:
                fl = {}
            o = obs["farms"][1 - me]
            self.live.push(int(self.prev_obs["step"]), fl, o, o["money"], obs["market"]["inventory"]["WHEAT"])
        rs, rd = self.live.rows()
        self.cur_rows = (torch.from_numpy(rs).view(1, 1, -1), torch.from_numpy(rd).view(1, 1, -1))

    @torch.inference_mode()
    def _feed(self, h):
        if self.cur_rows is not None:
            n = h.size(0)
            self.net.engram_ids = {LAYER_S: self.cur_rows[0].expand(1, n, -1), LAYER_D: self.cur_rows[1].expand(1, n, -1)}
            self.net.engram_mask = None
        return super()._feed(h)

    def __call__(self, obs, config=None):
        if int(obs["step"]) == 0 and self.pos:
            self.reset()
        self._sync(obs)
        action = super().__call__(obs, config)
        self.prev_obs, self.prev_action = copy.deepcopy(obs), copy.deepcopy(action)
        return action
