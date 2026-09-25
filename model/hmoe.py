"""Heuristic Mixture-of-Experts action layer (H-MoE), modelled on DeepSeek-V4.1-Flash `inference/model.py`
(Gate L792-827, Expert L830-851, MoE L854-904).

Routed experts are independent heuristic algorithms (agent/top_experts.py: strategies extracted from top teams'
replays; public agents; skill layers). As in DSV4.1 `MoE.forward` (`y += self.shared_experts(x)`), a SHARED
expert is always on and unweighted: here the strongest complete heuristic (cha22) plus the neural path. For a
decision slot with hidden state h:

  Gate (as DSV4.1):  scores  = sqrtsoftplus(W_g h / gate_temp)
                     indices = topk(scores + bias[slot_type])      # bias picks experts, never scales them;
                                                                   # one bias row per slot type (farmer / hand /
                                                                   # market), like DSV4.1's bias_vl for image spans
                     weights = scores.gather(indices), normalised if topk > 1, * route_scale
                     skills with no proposal for this slot are excluded from selection
  Routed expert k:   E_k = SwiGLU(action_enc(desc(a_k)) + expert_emb[k])   # a_k: skill k's proposed candidate
                     (one SwiGLU shared by all skills + a per-skill embedding: 157 fine-grained experts at the
                      cost of one; same clamps as DSV4.1's Expert)
  Shared expert:     S = SwiGLU(action_enc(desc(a_s)) + shared_emb) + SwiGLU_nn(h)   # a_s: shared heuristic
  Output (as MoE):   y = h + S + sum_k w_k E_k                             # into the CLM state head
  Pointer copy:      logits(a) += lam * ([a == a_s] + sum_k w_k [a == a_k])    # reproduce experts exactly
  Safe start:        the gate score has a learnable per-expert offset (init -6, the one addition to DSV4.1's gate)
                     so routed weights start ~0 and the policy starts as the shared expert.
Selection bias is updated outside the optimiser (noaux_tc, as `Transformer.update_gate_bias`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOT_TYPES = 3            # 0 farmer, 1 hand, 2 market (SLOT_FARMER, SLOT_HAND, SLOT_MARKET)


class HGate(nn.Module):
    def __init__(self, dim, n_experts, topk=(1, 1, 1), score_func="sqrtsoftplus", gate_temp=1.0,
                 route_scale=1.5, norm_topk_prob=True, score_offset=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.topk = tuple(topk)                                  # per slot type (cf. get_moe_config per layer)
        self.score_func, self.gate_temp = score_func, gate_temp
        self.route_scale, self.norm_topk_prob = route_scale, norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(n_experts, dim))
        self.score_offset = nn.Parameter(torch.full((n_experts,), float(score_offset)))
        self.register_buffer("bias", torch.zeros(N_SLOT_TYPES, n_experts))
        self.last_load = None

    def scores(self, x):
        s = (F.linear(x.float(), self.weight.float()) + self.score_offset.float()) / self.gate_temp
        if self.score_func == "softmax":
            return s.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return s.sigmoid()
        return F.softplus(s).sqrt()

    def forward(self, x, slot_type, avail):
        """x [n,dim]; slot_type [n] long; avail [n,E] bool -> weights [n,K], indices [n,K] (K = max topk;
        entries beyond a row's own topk get weight 0)."""
        scores = self.scores(x)
        sel = (scores + self.bias[slot_type]).masked_fill(~avail, float("-inf"))
        K = max(self.topk)
        indices = sel.topk(K, dim=-1)[1]
        weights = scores.gather(1, indices)
        k_row = torch.tensor(self.topk, device=x.device)[slot_type]
        keep = (torch.arange(K, device=x.device)[None] < k_row[:, None]) & avail.gather(1, indices)
        weights = weights * keep
        if self.norm_topk_prob:
            multi = (k_row > 1)[:, None]
            weights = torch.where(multi, weights / (weights.sum(-1, keepdim=True) + 1e-20), weights)
        weights = weights * self.route_scale
        with torch.no_grad():
            load = torch.zeros(N_SLOT_TYPES, self.n_experts, device=x.device)
            load.index_put_((slot_type[:, None].expand_as(indices)[keep], indices[keep]),
                            torch.ones_like(indices[keep], dtype=torch.float), accumulate=True)
            self.last_load = load
        return weights, indices

    @torch.no_grad()
    def update_bias(self, speed=1e-3):
        """noaux_tc balancing per slot type (only over experts that were available at all)."""
        if self.last_load is None:
            return
        for t in range(N_SLOT_TYPES):
            load = self.last_load[t]
            if load.sum() == 0:
                continue
            self.bias[t] += speed * torch.sign(load.mean() - load)


class SwiGLU(nn.Module):
    """DSV4.1 Expert: up clamped on both sides, gate only from above."""

    def __init__(self, dim, inter_dim, swiglu_limit=10.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(x.dtype))


class HeuristicMoE(nn.Module):
    def __init__(self, dim, n_experts, inter_dim=None, topk=(1, 1, 1), route_scale=1.5, pointer_init=60.0,
                 swiglu_limit=10.0, score_offset=-6.0):
        super().__init__()
        self.n_experts = n_experts
        self.gate = HGate(dim, n_experts, topk=topk, route_scale=route_scale, score_offset=score_offset)
        self.shared_emb = nn.Parameter(torch.zeros(dim))
        self.shared_heur = SwiGLU(dim, inter_dim or 2 * dim, swiglu_limit)
        self.expert_emb = nn.Embedding(n_experts, dim)
        nn.init.normal_(self.expert_emb.weight, std=0.02)
        self.routed = SwiGLU(dim, inter_dim or 2 * dim, swiglu_limit)
        self.shared = SwiGLU(dim, inter_dim or 2 * dim, swiglu_limit)
        nn.init.zeros_(self.routed.w2.weight)                     # start as the identity on h
        nn.init.zeros_(self.shared.w2.weight)
        nn.init.zeros_(self.shared_heur.w2.weight)
        self.log_lam = nn.Parameter(torch.tensor(math.log(pointer_init)))

    def lam(self):
        return self.log_lam.exp()

    def forward(self, h, slot_type, props, prop_enc, shared_prop=None):
        """h [n,dim]; slot_type [n]; props [n,E] long candidate per routed expert (-1: no proposal);
        shared_prop [n] candidate of the shared heuristic expert (-1: none);
        prop_enc: callable(cand_idx [m], slot_type [m]) -> action encodings [m,dim].
        Returns y [n,dim] (for the CLM state head), weights [n,K], chosen candidates [n,K] (-1 if none)."""
        avail = props >= 0
        weights, idx = self.gate(h, slot_type, avail)
        cand = props.gather(1, idx)
        ok = (cand >= 0) & (weights > 0)
        n, K = idx.shape
        e = torch.zeros(n, K, h.size(-1), dtype=h.dtype, device=h.device)
        if ok.any():
            rows = torch.arange(n, device=h.device)[:, None].expand(n, K)[ok]
            e[ok] = prop_enc(cand[ok], slot_type[rows]).to(h.dtype) + self.expert_emb(idx[ok]).to(h.dtype)
        routed = self.routed(e, weights.unsqueeze(-1).to(torch.float32)).sum(1)
        y = h + self.shared(h) + routed.to(h.dtype)
        self.last_shared = shared_prop
        if shared_prop is not None and (shared_prop >= 0).any():          # shared expert: always on, unweighted
            ok_s = shared_prop >= 0
            es = torch.zeros_like(h)
            es[ok_s] = (prop_enc(shared_prop[ok_s], slot_type[ok_s]).to(h.dtype) + self.shared_emb.to(h.dtype))
            y = y + self.shared_heur(es) * ok_s[:, None].to(h.dtype)
        self.last_idx = idx                                        # which skills were routed (for callers)
        return y, weights, torch.where(ok, cand, torch.full_like(cand, -1))

    def pointer_logits(self, logits, weights, cand, shared_prop=None):
        """logits [n,N] (one slot family) += lam * ([a == a_s] + sum_k w_k [a == a_k])."""
        bonus = torch.zeros_like(logits)
        ok = cand >= 0
        bonus.scatter_add_(1, cand.clamp(min=0), (weights * ok).to(logits.dtype))
        if shared_prop is not None:
            ok_s = shared_prop >= 0
            bonus[ok_s, shared_prop[ok_s]] += 1.0
        return logits + self.lam().to(logits.dtype) * bonus

    @torch.no_grad()
    def favour(self, expert, margin=1.0):
        """Safe initialisation: make `expert` the top-1 pick wherever it has a proposal."""
        self.gate.bias[:, expert] += margin
