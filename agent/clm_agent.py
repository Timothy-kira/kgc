"""Inference agent for the CLM-head decoder-only policy (torch, CPU, incremental KV caches).

Per game step:
  1. the 33 observation tokens + <ACT> are fed as one chunk
  2. slot by slot (farmer, each hand, market orders until STOP): state_head(hidden) is scored against the
     cached, projected candidate matrix (CLM: exp(logit_scale) * cos), masked, argmax/sampled, and the chosen
     decision (slot embedding + candidate encoding) is fed back
  3. decisions -> action dict (always a valid action)
Kaggle runtime: torch 2.6 CPU, 2 threads; actTimeout 1 s per step, runTimeout 1200 s per game.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.action_space import (MAX_ORDERS, SLOT_FARMER, SLOT_HAND, SLOT_MARKET, STOP, UNIT_NONE, SlotPlan,
                                cand_to_order, cand_to_unit, decisions_to_action)
from agent.skills import market_proposal, slot_proposals
from agent.features import Tracker
from agent.obs_tokens import OBS_TYPES, TYPE_OF_SLOT, build_step, quad_features
from model.clm_policy import ACT_ID, OBS_IDS, SLOT_IDS, CLMPolicy
from model.dsv41 import ModelArgs

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_clm(weights, args_json):
    d = json.load(open(args_json))
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    net = CLMPolicy(ModelArgs(**d))
    sd = torch.load(weights, map_location="cpu") if isinstance(weights, str) else weights
    net.load_state_dict(sd, strict=False)
    return net.eval()


class CLMAgent:
    def __init__(self, net, temperature=0.0, time_budget=0.85, seed=None, hier=None, skills=None,
                 favour=None, favour_margin=5.0):
        self.net = net
        # heuristic skill experts (agent/skills.SkillPool) routed by the H-MoE layer (model/hmoe.py)
        self.skills = skills if getattr(net, "hmoe", None) is not None else None
        if self.skills is not None and favour is not None:
            net.hmoe.favour(self.skills.names.index(favour), favour_margin)   # safe init: start as that skill
        # hierarchical greedy decoding: op -> item -> quantity by marginal probability (see _pick)
        self.hier = bool(int(os.environ.get("CLM_HIER", "0"))) if hier is None else hier
        self.temperature = temperature
        self.time_budget = time_budget
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.inference_mode():
            self.zu, self.zm = net.candidate_matrices()                     # cached projected candidates
            self.enc_u = net.action_enc(net.unit_desc)                      # decision input encodings
            self.enc_m = net.action_enc(net.market_desc)
            self.slot_emb = net.embed(torch.tensor(SLOT_IDS))
            self.obs_ids = torch.tensor([OBS_IDS[t] for t in TYPE_OF_SLOT])
            self.act_emb = net.embed(torch.tensor([ACT_ID]))
        self.desc_u = torch.as_tensor(np.asarray(net.unit_desc.cpu()), dtype=torch.long)
        self.desc_m = torch.as_tensor(np.asarray(net.market_desc.cpu()), dtype=torch.long)
        self.reset()

    def reset(self):
        self.caches = self.net.init_cache(1)
        self.pos = 0
        self.tracker = Tracker()
        self.times = []
        self.trace = []

    @torch.inference_mode()
    def _obs_embeddings(self, obs):
        self.tracker.update(obs)
        P, G, _, _ = self.tracker.features(None)
        st = build_step(obs, P, G)
        net = self.net
        h = net.embed(self.obs_ids)
        feats = [torch.from_numpy(st["prod"].astype(np.float32)), torch.from_numpy(st["glob"].astype(np.float32))[None],
                 quad_features(torch.from_numpy(st["tiles"].astype(np.float32) / 255.0)[None])[0],
                 torch.from_numpy(st["units"].astype(np.float32))]
        rows, idx = [], [0, 0, 0, 0]
        proj = [net.obs_proj[t](feats[t]) for t in range(4)]
        for t in TYPE_OF_SLOT:
            rows.append(proj[t][idx[t]])
            idx[t] += 1
        return h + torch.stack(rows)

    @torch.inference_mode()
    def _feed(self, h):
        out = self.net.step(h.unsqueeze(0), self.pos, self.caches)
        self.pos += h.size(0)
        return out[0, -1]

    @torch.inference_mode()
    def __call__(self, obs, config=None):
        t0 = time.time()
        if int(obs["step"]) == 0 and self.pos:
            self.reset()
        net = self.net
        n_hands = len(obs["farms"][int(obs["player"])]["hands"])
        skill = self._skill_proposals(obs, config, n_hands) if self.skills is not None else None
        h = self._feed(torch.cat([self._obs_embeddings(obs), self.act_emb], 0))
        plan = SlotPlan(n_hands)
        decisions = []
        raw = []                                   # exact skill item when the choice equals a routed proposal
        scale = float(net.scale())
        while not plan.done:
            if skill is not None:
                slot, mask = plan.mask()
                c, r = self._hmoe_slot(h, slot, mask, skill, sum(1 for d in decisions if d[0] == SLOT_MARKET),
                                       len([d for d in decisions if d[0] != SLOT_MARKET]),
                                       time.time() - t0 > self.time_budget)
                enc = (self.enc_m if slot == SLOT_MARKET else self.enc_u)[c]
                plan.feed(slot, c)
                decisions.append((slot, c))
                raw.append(r)
                h = self._feed((self.slot_emb[min(slot, 2)] + enc).unsqueeze(0))
                continue
            slot, mask = plan.mask()
            late = time.time() - t0 > self.time_budget
            zs = F.normalize(net.state_head(h).float(), dim=-1)
            if slot == SLOT_MARKET:
                if late:                                           # out of time: close the order list
                    c = STOP
                else:
                    lg = scale * (self.zm @ zs)
                    if mask is not None:
                        lg = lg.masked_fill(~torch.from_numpy(mask), -1e9)
                    c = self._pick(lg, self.desc_m)
                enc = self.enc_m[c]
            else:
                if late:
                    c = UNIT_NONE
                else:
                    c = self._pick(scale * (self.zu @ zs), self.desc_u)
                enc = self.enc_u[c]
            plan.feed(slot, c)
            decisions.append((slot, c))
            h = self._feed((self.slot_emb[min(slot, 2)] + enc).unsqueeze(0))
        self.times.append(time.time() - t0)
        self.trace.append(decisions)
        if skill is not None:
            return self._build_action(decisions, raw)
        return decisions_to_action(decisions)

    # ------------------------------------------------------------------ heuristic MoE path
    def _skill_proposals(self, obs, config, n_hands):
        acts = [a for _, a in self.skills.propose(obs, config)]
        units, markets = slot_proposals(acts, n_hands)
        raw_u, raw_m = [], []
        for a in acts:
            hands = a.get("hands") if isinstance(a.get("hands"), list) else []
            raw_u.append([a.get("farmer")] + [hands[i] if i < len(hands) else ["PASS"] for i in range(n_hands)])
            m = a.get("market") if isinstance(a.get("market"), list) else []
            raw_m.append(m[:MAX_ORDERS])
        return dict(units=units, markets=markets, raw_u=raw_u, raw_m=raw_m)

    def _hmoe_slot(self, h, slot, mask, sk, j_market, pos_unit, late):
        net = self.net
        if slot == SLOT_MARKET:
            props = market_proposal(sk["markets"], j_market)
            z, desc = self.zm, self.desc_m
        else:
            props = [u[pos_unit] if pos_unit < len(u) else UNIT_NONE for u in sk["units"]]
            z, desc = self.zu, self.desc_u
        props_t = torch.tensor(props, dtype=torch.long)[None]
        lg, w, cand, idx = net.hmoe_logits(h[None], torch.tensor([slot]), props_t, z)
        lg, w, cand, idx = lg[0], w[0], cand[0], idx[0]
        if mask is not None:
            lg = lg.masked_fill(~torch.from_numpy(mask), -1e9)
        if late and (cand >= 0).any():                          # out of time: follow the top-weighted skill
            c = int(cand[int(torch.where(cand >= 0, w, torch.full_like(w, -1)).argmax())])
        else:
            c = self._pick(lg, desc)
        r = None                                                # exact item of a routed skill that proposed c
        for k_i, cc, ww in zip(idx.tolist(), cand.tolist(), w.tolist()):
            if cc == c and ww > 0:
                if slot == SLOT_MARKET:
                    lst = sk["raw_m"][k_i]
                    r = lst[j_market] if j_market < len(lst) else None
                else:
                    r = sk["raw_u"][k_i][pos_unit]
                break
        return c, r

    @staticmethod
    def _build_action(decisions, raw):
        act = {"farmer": ["PASS"], "hands": [], "market": []}
        for (slot, c), r in zip(decisions, raw):
            if slot == SLOT_FARMER:
                act["farmer"] = r if isinstance(r, list) and r else (cand_to_unit(c) or ["PASS"])
            elif slot == SLOT_HAND:
                act["hands"].append(r if isinstance(r, list) and r else (cand_to_unit(c) or ["PASS"]))
            elif c != STOP:
                act["market"].append(r if isinstance(r, list) else cand_to_order(c))
        return act

    def _pick(self, lg, desc=None):
        if self.temperature > 0:
            return int(torch.multinomial(torch.softmax(lg / self.temperature, -1), 1, generator=self.gen))
        if not self.hier or desc is None:
            return int(lg.argmax())
        # Plain argmax over op x item x qty favours a candidate whose mass is concentrated on one exact form (a
        # fixed filler order) over an intent whose mass is split across many quantities/items. Choose the op by
        # marginal probability, then the item within it, then the best candidate.
        p = torch.softmax(lg.float(), -1)
        sel = torch.ones_like(p, dtype=torch.bool)
        for col in (1, 2):
            ids = desc[:, col]
            marg = torch.zeros(int(ids.max()) + 1).index_add_(0, ids[sel], p[sel])
            sel &= ids == int(marg.argmax())
        return int(torch.where(sel, p, torch.full_like(p, -1.0)).argmax())
