"""Decoder-only policy (DeepSeek-V4.1 CSA2 backbone) with CLM-style constrained action heads.

Action heads follow Stanford/Hazy Research CLM (github.com/Contrastive-LM/CLM, src/clm/heads.py):
  * `make_head`: hidden -> width -> (depth-2 hidden blocks, optional LayerNorm / residual) -> proj MLP
  * a state head (on the backbone hidden state that must make the decision) and an action head (on the
    candidate action embedding), both L2-normalised
  * score = exp(logit_scale) * cos(state, action), logit_scale init log(1/0.07), exp clamped at 100
  * training: cross-entropy over the closed candidate set of the slot ("choice" objective)
Candidates (agent/action_space.py) are described by (kind, op, item, qty) and embedded by a small action
encoder; at inference the projected candidate matrices are cached (one dot product per candidate).

Sequence per step:  [33 observation tokens][<ACT>][d_1][d_2]...[d_m]
where d_j are decision tokens (farmer, hands..., market orders..., STOP). Decision j is predicted from the
hidden state of the token before it; its input embedding = slot-type embedding + candidate encoding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.action_space import (ITEM_VOCAB, MARKET_DESC, N_MARKET, N_UNIT, OP_VOCAB, QTY_VOCAB, UNIT_DESC)
from model.dsv41 import Block, ModelArgs, Transformer, make_identity_pre_mix

# token ids of the (small) input vocabulary
OBS_IDS = [0, 1, 2, 3]
ACT_ID = 4
SLOT_IDS = [5, 6, 7]          # farmer, hand, market
PAD_ID = 8
VOCAB = 16


def make_head(hidden, width, depth=2, proj=128, activation="gelu", layernorm=False, residual=False):
    """CLM heads.py make_head: hidden -> width -> ... -> proj."""
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class ActionEncoder(nn.Module):
    """(kind, op, item, qty) description -> dim-d candidate embedding."""

    def __init__(self, dim):
        super().__init__()
        self.kind = nn.Embedding(2, dim)
        self.op = nn.Embedding(len(OP_VOCAB), dim)
        self.item = nn.Embedding(len(ITEM_VOCAB), dim)
        self.qty = nn.Embedding(len(QTY_VOCAB), dim)
        q = torch.zeros(len(QTY_VOCAB))
        for i, s in enumerate(QTY_VOCAB):
            if s.isdigit():
                q[i] = math.log1p(int(s)) / math.log1p(9999)
        self.register_buffer("qnum", q, persistent=False)
        self.qlin = nn.Linear(1, dim)
        self.mix = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        for e in (self.kind, self.op, self.item, self.qty):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, desc):
        e = self.kind(desc[..., 0]) + self.op(desc[..., 1]) + self.item(desc[..., 2]) + self.qty(desc[..., 3]) \
            + self.qlin(self.qnum[desc[..., 3]].unsqueeze(-1))
        return e + self.mix(e)


class CLMPolicy(Transformer):
    def __init__(self, args: ModelArgs, head_width=256, head_depth=3, proj=128):
        args.vocab_size = VOCAB
        super().__init__(args)
        del self.head                                          # no open-vocabulary LM head
        self.action_enc = ActionEncoder(args.dim)
        # CLM training defaults: gelu, layernorm=True, residual=False, depth 3
        self.state_head = make_head(args.dim, head_width, head_depth, proj, "gelu", True, False)
        self.action_head = make_head(args.dim, head_width, head_depth, proj, "gelu", True, False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.register_buffer("unit_desc", torch.from_numpy(UNIT_DESC), persistent=False)
        self.register_buffer("market_desc", torch.from_numpy(MARKET_DESC), persistent=False)
        self._cand_cache = None
        # CLM-style opponent event head (docs/PLAN_v4.1.md): per-horizon query vs encoded candidate events
        from agent.opp_events import HORIZONS, DESC_DIM, PRODUCTS, opp_desc
        self.opp_q = nn.ModuleList([make_head(args.dim, head_width, head_depth, proj, "gelu", True, False)
                                    for _ in HORIZONS])
        self.opp_cand = make_head(DESC_DIM, head_width, head_depth, proj, "gelu", True, False)
        self.opp_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.opp_stock = nn.Sequential(nn.Linear(args.dim, args.dim), nn.GELU(), nn.Linear(args.dim, len(PRODUCTS)))
        self.register_buffer("opp_desc", torch.from_numpy(opp_desc()), persistent=False)
        self.last_opp = None
        self.hmoe = None
        if getattr(args, "n_skills", 0):
            from model.hmoe import HeuristicMoE
            self.hmoe = HeuristicMoE(args.dim, args.n_skills, topk=tuple(args.skill_topk))

    # ------------------------------------------------------------------ candidates
    def candidate_matrices(self, use_cache=False):
        """L2-normalised projected candidates: unit [N_UNIT,P], market [N_MARKET,P]."""
        if use_cache and self._cand_cache is not None:
            return self._cand_cache
        zu = F.normalize(self.action_head(self.action_enc(self.unit_desc)).float(), dim=-1)
        zm = F.normalize(self.action_head(self.action_enc(self.market_desc)).float(), dim=-1)
        if use_cache:
            self._cand_cache = (zu, zm)
        return zu, zm

    def clear_cache(self):
        self._cand_cache = None

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

    def score(self, h, kind, zu, zm):
        """h [N,dim] hidden states; kind [N] (0 unit, 1 market) -> (unit logits [Nu,N_UNIT], market logits [Nm,N_MARKET])."""
        zs = F.normalize(self.state_head(h).float(), dim=-1)
        s = self.scale()
        lu = s * zs[kind == 0] @ zu.t()
        lm = s * zs[kind == 1] @ zm.t()
        return lu, lm

    # ------------------------------------------------------------------ heuristic MoE (model/hmoe.py)
    def prop_encode(self, cand, slot_type):
        """Action encodings of skill-proposed candidates (unit candidates for farmer/hand, market otherwise)."""
        out = torch.empty(len(cand), self.args.dim, device=cand.device)
        mk = slot_type == 2
        if (~mk).any():
            out[~mk] = self.action_enc(self.unit_desc[cand[~mk]]).float()
        if mk.any():
            out[mk] = self.action_enc(self.market_desc[cand[mk]]).float()
        return out

    def hmoe_logits(self, h, slot_type, props, z, shared_prop=None):
        """h [n,dim] slot hidden states of ONE candidate family (z = zu or zm); props [n,E] skill proposals
        (-1 = none). -> logits [n,N] = CLM score of the H-MoE output + pointer copy, gate weights, candidates."""
        y, w, cand = self.hmoe(h, slot_type, props, self.prop_encode, shared_prop)
        zs = F.normalize(self.state_head(y).float(), dim=-1)
        logits = self.scale() * zs @ z.t()
        return self.hmoe.pointer_logits(logits, w, cand, shared_prop), w, cand, self.hmoe.last_idx

    def opp_forward(self, h):
        """h [n, dim] hidden states at <ACT> -> (logits [n, H, 9 products, 9 buckets], log1p stock [n, 9])."""
        z = F.normalize(self.opp_cand(self.opp_desc).float(), dim=-1)             # [H, 9, 9, P]
        s = self.opp_scale.exp().clamp(max=100.0)
        q = torch.stack([F.normalize(head(h).float(), dim=-1) for head in self.opp_q], 1)   # [n, H, P]
        return s * torch.einsum("nhp,hjbp->nhjb", q, z), self.opp_stock(h.float())

    def decision_embedding(self, slot, desc):
        return self.embed(torch.tensor(SLOT_IDS, device=desc.device)[slot]) + self.action_enc(desc)

    # ------------------------------------------------------------------ training
    def embed_all(self, ids, obs_feats, obs_type, dec_pos, dec_slot, dec_desc, dec_keep=None):
        h = self.embed_inputs(ids, obs_feats, obs_type)
        if dec_pos is not None and dec_pos.numel():
            add = self.action_enc(dec_desc).to(h.dtype)
            if dec_keep is not None:            # history dropout: hide which candidate was chosen (slot token stays)
                add = add * dec_keep.to(h.dtype).unsqueeze(-1)
            h = h.index_put((dec_pos[:, 0], dec_pos[:, 1]), h[dec_pos[:, 0], dec_pos[:, 1]] + add)
        return h

    def forward_train(self, ids, obs_feats, obs_type, dec_pos, dec_slot, dec_desc, tgt_pos, tgt_kind, val_pos,
                      mtp_pos=None, mtp_kind=None, dec_keep=None, opp_pos=None):
        h0 = self.embed_all(ids, obs_feats, obs_type, dec_pos, dec_slot, dec_desc, dec_keep)
        sh = self.shared
        sh.compress_kv = sh.index_k = sh.topk_idxs = sh.index_scores = None
        h = self.backbone(h0, checkpoint=True)
        zu, zm = self.candidate_matrices()
        lu, lm = self.score(h[tgt_pos[:, 0], tgt_pos[:, 1]], tgt_kind, zu, zm)
        value = self.value_head(h[val_pos[:, 0], val_pos[:, 1]].float())
        self.last_opp = self.opp_forward(h[opp_pos[:, 0], opp_pos[:, 1]]) if opp_pos is not None else None
        aux = self.last_aux
        mtp = None
        if self.mtp and mtp_pos is not None and mtp_pos.numel():
            m = self.mtp[0]
            e = torch.roll(h0, -1, 1)                           # input embedding of the next token
            x = m.proj(torch.cat([m.hnorm(h), m.enorm(e)], -1))
            x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
            x, pm = m.block(x, make_identity_pre_mix(x, self.hc_mult))
            x = m.norm(m.block.hc_pre(x, pm))
            mtp = self.score(x[mtp_pos[:, 0], mtp_pos[:, 1]], mtp_kind, zu, zm)
        return lu, lm, value, mtp, aux

    def forward(self, *args, **kw):
        return self.forward_train(*args, **kw)
