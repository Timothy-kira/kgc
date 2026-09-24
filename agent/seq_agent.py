"""Inference agent for the decoder-only policy (torch, CPU, incremental KV caches).

Per game step:
  1. build the 33 observation tokens and feed them (plus <ACT>) through `Transformer.step`
  2. decode action tokens with grammar masking (only legal next tokens) until <EOS>
  3. turn the tokens back into the action dict (exact inverse of the tokenizer)
The Kaggle runtime provides torch 2.6 (CPU, 2 threads), verified by a probe submission.
"""
import json
import time

import numpy as np
import torch

from agent.action_tokens import TOK, Grammar, decode
from agent.features import Tracker
from agent.obs_tokens import N_OBS, OBS_TOKEN_ID, OBS_TYPES, TYPE_OF_SLOT, build_step, quad_features
from model.dsv41 import ModelArgs, Transformer

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_model(weights, args_json=None, args=None):
    if args is None:
        d = json.load(open(args_json))
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        args = ModelArgs(**d)
    net = Transformer(args)
    sd = torch.load(weights, map_location="cpu") if isinstance(weights, str) else weights
    net.load_state_dict(sd, strict=False)
    return net.eval()


class SeqAgent:
    def __init__(self, net, temperature=0.0, allow_base=False, max_tokens=90, time_budget=0.8, seed=None):
        self.net = net
        self.temperature = temperature
        self.allow_base = allow_base
        self.max_tokens = max_tokens
        self.time_budget = time_budget
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.reset()

    def reset(self):
        self.caches = self.net.init_cache(1)
        self.pos = 0
        self.tracker = Tracker()
        self.trace = []            # (token ids, logprobs) per step, for RL

    @torch.inference_mode()
    def _obs_embeddings(self, obs):
        self.tracker.update(obs)
        P, G, _, _ = self.tracker.features(None)
        st = build_step(obs, P, G)
        ids = torch.tensor([OBS_TOKEN_ID[t] for t in TYPE_OF_SLOT], dtype=torch.long).view(1, -1)
        h = self.net.embed(ids)
        feats = [torch.from_numpy(st["prod"].astype(np.float32)), torch.from_numpy(st["glob"].astype(np.float32))[None],
                 quad_features(torch.from_numpy(st["tiles"].astype(np.float32) / 255.0)[None])[0],
                 torch.from_numpy(st["units"].astype(np.float32))]
        rows = []
        idx = {0: 0, 1: 0, 2: 0, 3: 0}
        for slot, t in enumerate(TYPE_OF_SLOT):
            rows.append(self.net.obs_proj[t](feats[t][idx[t]]))
            idx[t] += 1
        return h + torch.stack(rows)[None]

    @torch.inference_mode()
    def _feed(self, h):
        """Feed embeddings h [1,n,dim] as one chunk; returns the hidden states."""
        out = self.net.step(h, self.pos, self.caches)
        self.pos += h.size(1)
        return out

    @torch.inference_mode()
    def __call__(self, obs, config=None):
        t0 = time.time()
        if int(obs["step"]) == 0 and self.pos:
            self.reset()
        h_obs = self._obs_embeddings(obs)
        n_hands = len(obs["farms"][int(obs["player"])]["hands"])
        g = Grammar(max_hands=n_hands, allow_base=self.allow_base)
        toks, lps = [], []
        tok = TOK["<ACT>"]
        g.feed(tok)
        toks.append(tok)
        # observation tokens + <ACT> in one chunk
        h = self._feed(torch.cat([h_obs, self.net.embed(torch.tensor([[tok]]))], 1))
        while not g.done and len(toks) < self.max_tokens:
            logits = self.net.head(h[0, -1].float())
            m = g.mask()
            mask = torch.zeros(logits.numel(), dtype=torch.bool)
            mask[:len(m)] = torch.from_numpy(m)                  # observation-type ids are never generated
            if time.time() - t0 > self.time_budget:      # out of time: close the action as fast as possible
                pref = [TOK["<EOS>"], TOK["<MARKET>"], TOK["<NONE>"], TOK["<DEFAULT>"]]
                for p in pref:
                    if mask[p]:
                        logits = torch.full_like(logits, -1e9)
                        logits[p] = 0
                        break
            logits = logits.masked_fill(~mask, -1e9)
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, -1)
                tok = int(torch.multinomial(probs, 1, generator=self.gen))
                lps.append(float(torch.log(probs[tok] + 1e-12)))
            else:
                tok = int(logits.argmax())
                lps.append(0.0)
            g.feed(tok)
            toks.append(tok)
            h = self._feed(self.net.embed(torch.tensor([[tok]])))
        self.trace.append((toks, lps))
        try:
            act = decode(toks)
            return PASS if act == "BASE" else act
        except Exception:
            return PASS
