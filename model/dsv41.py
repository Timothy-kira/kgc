"""Decoder-only policy for Kaggriculture, ported from DeepSeek-V4.1-Flash `inference/model.py`.

Faithful to the reference (same class / argument names and forward logic) for:
  * ModelArgs field names, RMSNorm, precompute_freqs_cis (RoPE, optional YaRN), apply_rotary_emb
  * Attention: low-rank Q (wq_a -> q_norm -> wq_b), single shared latent KV head (wkv -> kv_norm),
    RoPE on the last rope_head_dim channels, learnable attention sink, output de-rotation,
    grouped low-rank output projection (wo_a block-diagonal over o_groups, then wo_b)
  * CSA2 sparse attention: every layer attends to a sliding window of raw KV; layers with
    compress_ratio > 0 additionally attend to `index_topk` compressed positions picked by an Indexer.
    Only kv_source_layers run a Compressor; only index_source_layers run an Indexer; the other layers
    reuse the shared compressed KV / top-k indices (Full / Reindex / Reuse modes)
  * Compressor: gated softmax pooling of `compress_ratio` tokens into one KV latent (ratio 1 = projection)
  * Indexer: relu(q . k) weighted by weights_proj, top-k over visible compressed positions
  * MoE: sqrtsoftplus gate with selection bias, top-k routed SwiGLU experts + one shared expert
  * Hyper-Connections (hc_mult residual copies, Sinkhorn-normalised comb, pre-mix handed to next sublayer)
  * MTP head for multi-token prediction (DeepSeek-V3 style: next hidden + embedding of the next token)
Left out (not needed at this scale / on CPU): fp8/fp4 quantisation, tensor parallelism, Engram, vision,
DSpark speculative head, the hierarchical candidate pre-filter.

Additions needed for *training* (the reference is inference-only):
  * autograd-friendly (non in-place) RoPE and a chunked gather implementation of `sparse_attn`
  * DeepSeek-V3.2 style indexer training: KL(main attention mass on the selected compressed positions
    || indexer distribution), the attention target detached
  * observation embeddings: the sequence interleaves per-step observation tokens (continuous features
    projected into the embedding, like image patch tokens in the VL model) with action tokens

Two execution modes, mirroring the reference:
  forward_train(...)          full sequence, start_pos = 0 (prefill-like), with gradients
  step(...) / KV caches       incremental decoding at inference (start_pos > 0 paths)
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from agent.action_tokens import V as ACTION_VOCAB


# ------------------------------------------------------------------------------------------ args
@dataclass
class ModelArgs:
    max_batch_size: int = 1
    max_seq_len: int = 49152
    vocab_size: int = ACTION_VOCAB + 8          # action tokens + observation type tokens
    dim: int = 128
    moe_inter_dim: int = 256
    n_layers: int = 6
    n_mtp_layers: int = 1
    n_heads: int = 4
    # moe
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 1
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 10.0
    # attention
    q_lora_rank: int = 96
    head_dim: int = 64
    rope_head_dim: int = 16
    norm_eps: float = 1e-6
    o_groups: int = 2
    o_lora_rank: int = 64
    window_size: int = 128
    # one entry per layer, MTP layers included: 0 = sliding window only, r = KV compressed r-to-1
    compress_ratios: tuple = (0, 2, 2, 1, 1, 0, 0)
    kv_source_layers: tuple = (1, 3)
    index_source_layers: tuple = (1, 3)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 64
    # hyper-connections
    hc_mult: int = 2
    hc_sinkhorn_iters: int = 5
    hc_eps: float = 1e-6
    # observation embedding (per type: feature width)
    obs_types: tuple = ()           # filled by the data layout (see agent/obs_tokens.py)
    # value head
    n_value: int = 2
    # training
    attn_chunk: int = 2048
    index_loss_coef: float = 0.1


# ------------------------------------------------------------------------------------------ basics
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis, inverse=False):
    """Non in-place version of the reference (returns the rotated tensor). x: [b,s,d] or [b,s,h,d]."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(dtype)


def rope_tail(x, freqs_cis, rd, inverse=False):
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)], -1)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=2048, return_probs=False):
    """Reference semantics of kernel.sparse_attn: for each query gather kv rows by index (-1 = none),
    softmax over them plus a per-head sink logit (the sink takes mass but contributes no value).
    q [b,m,h,d], kv [b,n,d], topk_idxs [b,m,k] int. Chunked over queries to bound memory."""
    b, m, h, d = q.shape
    outs, probs = [], []
    for s in range(0, m, chunk):
        qi = q[:, s:s + chunk]
        idx = topk_idxs[:, s:s + chunk].long()
        valid = idx >= 0
        g = torch.gather(kv, 1, idx.clamp(min=0).flatten(1).unsqueeze(-1).expand(-1, -1, d)).view(b, idx.size(1), idx.size(2), d)
        sc = torch.einsum("bmhd,bmkd->bmhk", qi, g.to(qi.dtype)).float() * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), -1e30)
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, sc.size(1), h, 1)
        p = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]
        outs.append(torch.einsum("bmhk,bmkd->bmhd", p.to(g.dtype), g).to(q.dtype))
        if return_probs:
            probs.append(p)
    o = torch.cat(outs, 1)
    return (o, torch.cat(probs, 1)) if return_probs else o


def window_idxs(window_size, seqlen, device):
    """Training (prefill) version of get_window_topk_idxs: row t -> positions t-W+1..t (-1 before 0)."""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
    return torch.where(idxs > end, -1, idxs)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__(in_features, out_features, bias=bias)


class SharedAttentionRuntime:
    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.index_scores = None


# ------------------------------------------------------------------------------------------ CSA2
class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate."""

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.compress_ratio = args.compress_ratios[layer_id]
        self.head_dim = args.head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wkv = Linear(args.dim, args.head_dim)
        if self.compress_ratio > 1:
            self.wgate = Linear(args.dim, args.head_dim)

    def forward(self, x, start_pos=0, state=None):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x))
        xf = x.float()
        kv, score = self.wkv(xf), self.wgate(xf)
        if state is None:                      # prefill / training: pool complete groups
            cutoff = seqlen - seqlen % ratio
            if cutoff == 0:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
            score = score[:, :cutoff].unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            return self.norm(kv.to(x.dtype))
        # decode: one token per step, pool only when the group just completed
        slot = start_pos % ratio
        state["kv"][:bsz, slot] = kv.squeeze(1)
        state["score"][:bsz, slot] = score.squeeze(1)
        if (start_pos + 1) % ratio != 0:
            return None
        pooled = (state["kv"][:bsz] * state["score"][:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        return self.norm(pooled.to(x.dtype))


class Indexer(nn.Module):
    """Keeps the `index_topk` best compressed positions per query (relu(q.k) weighted by weights_proj)."""

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = min(args.rope_head_dim, args.index_head_dim)
        self.index_topk = args.index_topk
        self.softmax_scale = self.index_head_dim ** -0.5
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = Linear(args.dim, self.n_heads)
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)

    def make_k(self, latent, freqs):
        k = self.k_norm(self.wk(latent))
        return rope_tail(k, freqs, self.rope_head_dim)

    def qw(self, x, qr, freqs_q):
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = rope_tail(q, freqs_q, self.rope_head_dim)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        return q, weights

    def scores(self, x, qr, index_k, freqs_q):
        q, weights = self.qw(x, qr, freqs_q)
        s = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=2)          # [b,s,t]

    def selected_scores(self, x, qr, index_k, freqs_q, idx):
        """Differentiable indexer scores at the selected positions only (idx [b,s,k], -1 = none)."""
        q, weights = self.qw(x, qr, freqs_q)
        b, s_, k = idx.shape
        kk = torch.gather(index_k, 1, idx.clamp(min=0).flatten(1).unsqueeze(-1).expand(-1, -1, index_k.size(-1)))
        kk = kk.view(b, s_, k, -1)
        sc = torch.einsum("bshd,bskd->bshk", q.float(), kk.float())
        return (sc.relu() * weights.float().unsqueeze(-1)).sum(dim=2)          # [b,s,k]


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, shared: SharedAttentionRuntime, is_backbone=True):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.shared = shared
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = Linear(self.dim, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        self.wo_a = nn.Parameter(torch.empty(self.n_groups, self.o_lora_rank, self.n_heads * self.head_dim // self.n_groups))
        nn.init.normal_(self.wo_a, std=(self.n_heads * self.head_dim // self.n_groups) ** -0.5)
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor = Compressor(args, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id) if self.is_index_source else None
        if self.compress_ratio:
            orig, theta = args.original_seq_len, args.compress_rope_theta
        else:
            orig, theta = 0, args.rope_theta
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, orig, theta,
                                                               args.rope_factor, args.beta_fast, args.beta_slow),
                             persistent=False)
        self.aux_loss = None

    # ----- output projection (grouped low rank), as in the reference
    def _out(self, o, freqs):
        bsz, seqlen = o.shape[:2]
        o = rope_tail(o, freqs, self.rope_head_dim, inverse=True)
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a)
        return self.wo_b(o.flatten(2))

    # ----- training / prefill over a whole sequence (start_pos = 0)
    def forward(self, x):
        bsz, seqlen, _ = x.size()
        freqs = self.freqs_cis[:seqlen]
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = rope_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), freqs, rd)
        kv = rope_tail(self.kv_norm(self.wkv(x)), freqs, rd)
        idxs = window_idxs(self.window_size, seqlen, x.device).unsqueeze(0).expand(bsz, -1, -1)
        n_win_keys = kv.size(1)
        comp_idx = None
        if self.compress_ratio:
            ratio = self.compress_ratio
            if self.is_kv_source:
                latent = self.compressor(x)
                if latent is not None:
                    # group j takes the position of its first token, j * ratio
                    cf = self.freqs_cis[: seqlen - seqlen % ratio: ratio]
                    if self.indexer is not None:
                        self.shared.index_k = self.indexer.make_k(latent, cf)
                    self.shared.compress_kv = rope_tail(latent, cf, rd)
                else:
                    self.shared.compress_kv = None
            ckv = self.shared.compress_kv
            if ckv is not None and ckv.size(1) > 0:
                n_comp = ckv.size(1)
                compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
                if self.is_index_source:
                    tops = []
                    with torch.no_grad():                               # chunked top-k selection
                        for s in range(0, seqlen, self.args.attn_chunk):
                            e = min(seqlen, s + self.args.attn_chunk)
                            si = self.indexer.scores(x[:, s:e], qr[:, s:e], self.shared.index_k, freqs[s:e])
                            vis = torch.arange(n_comp, device=x.device) >= compress_lens[s:e]
                            si = si.masked_fill(vis, -torch.inf)
                            k = min(self.indexer.index_topk, n_comp)
                            top = si.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
                            ok = torch.gather(si, -1, top) > -torch.inf
                            tops.append(torch.where(ok, top, -1))
                    top = torch.cat(tops, 1)
                    self.shared.topk_idxs = top
                    if self.training:
                        self.shared.index_scores = self.indexer.selected_scores(x, qr, self.shared.index_k, freqs, top)
                comp_idx = self.shared.topk_idxs
                kv = torch.cat([kv, ckv], 1)
                idxs = torch.cat([idxs, torch.where(comp_idx >= 0, comp_idx + n_win_keys, -1)], -1)
        need_probs = self.training and self.is_index_source and comp_idx is not None
        res = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale, self.args.attn_chunk, need_probs)
        if need_probs:
            o, p = res
            # DeepSeek-V3.2 indexer training: KL(attention mass on selected compressed positions || indexer)
            w = self.window_size if seqlen >= self.window_size else seqlen
            tgt = p[..., w:].sum(2).detach()                                  # [b,s,k] over heads
            valid = comp_idx >= 0
            tgt = tgt * valid
            tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-9)
            logit = self.shared.index_scores.masked_fill(~valid, -1e9)
            lp = torch.log_softmax(logit, -1)
            has = valid.any(-1)
            kl = (tgt * (torch.log(tgt.clamp(min=1e-9)) - lp)).sum(-1)
            self.aux_loss = (kl * has).sum() / has.sum().clamp(min=1)
        else:
            o = res
            self.aux_loss = None
        return self._out(o, freqs)

    # ----- incremental decode (start_pos > 0), with explicit caches like the reference
    def init_cache(self, bsz, device):
        c = {"window": torch.zeros(bsz, self.window_size, self.head_dim, device=device), "pos": 0}
        if self.is_kv_source:
            n = self.args.max_seq_len // self.compress_ratio
            c["compress_kv"] = torch.zeros(bsz, n, self.head_dim, device=device)
            if self.compress_ratio > 1:
                c["state"] = {"kv": torch.zeros(bsz, self.compress_ratio, self.head_dim, device=device),
                              "score": torch.full((bsz, self.compress_ratio, self.head_dim), -torch.inf, device=device)}
            if self.indexer is not None:
                c["index_k"] = torch.zeros(bsz, n, self.indexer.index_head_dim, device=device)
        return c

    def decode(self, x, start_pos, cache):
        """x [b,1,dim] for position start_pos."""
        bsz = x.size(0)
        rd = self.rope_head_dim
        freqs = self.freqs_cis[start_pos:start_pos + 1]
        qr = self.q_norm(self.wq_a(x))
        q = rope_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), freqs, rd)
        kv = rope_tail(self.kv_norm(self.wkv(x)), freqs, rd)
        win = self.window_size
        cache["window"][:bsz, start_pos % win] = kv.squeeze(1)
        oldest = start_pos % win + 1
        widx = torch.cat([torch.arange(oldest, win), torch.arange(oldest)]).to(x.device)
        widx = torch.where(widx > start_pos, -1, widx)
        keys = cache["window"][:bsz]
        idxs = widx.view(1, 1, -1).expand(bsz, 1, -1)
        if self.compress_ratio:
            ratio = self.compress_ratio
            if self.is_kv_source:
                latent = self.compressor(x, start_pos, cache.get("state")) if ratio > 1 else self.compressor(x)
                if latent is not None:
                    j = start_pos // ratio
                    cf = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
                    if self.indexer is not None:
                        cache["index_k"][:bsz, j:j + 1] = self.indexer.make_k(latent, cf)
                    cache["compress_kv"][:bsz, j:j + 1] = rope_tail(latent, cf, rd)
                self.shared.compress_kv = cache["compress_kv"]
                if self.indexer is not None:
                    self.shared.index_k = cache["index_k"]
            n_comp = (start_pos + 1) // ratio
            if n_comp > 0:
                if self.is_index_source:
                    si = self.indexer.scores(x, qr, self.shared.index_k[:bsz, :n_comp], freqs)   # [b,1,n]
                    k = min(self.indexer.index_topk, n_comp)
                    top = si.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
                    self.shared.topk_idxs = top
                comp = self.shared.topk_idxs
                keys = torch.cat([keys, self.shared.compress_kv[:bsz, :n_comp]], 1)
                idxs = torch.cat([idxs, comp + win], -1)
        o = sparse_attn(q, keys, self.attn_sink, idxs, self.softmax_scale)
        return self._out(o, freqs)


# ------------------------------------------------------------------------------------------ MoE
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.dim) * args.dim ** -0.5)
        # selection bias (aux-loss-free balancing, noaux_tc): updated outside the optimiser
        self.register_buffer("bias", torch.zeros(args.n_routed_experts))

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


class Expert(nn.Module):
    def __init__(self, dim, inter_dim, swiglu_limit=0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x, weights=None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
                                      for _ in range(args.n_routed_experts)])
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.last_load = None

    def forward(self, x):
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        self.last_load = counts.detach()
        for i in range(self.n_routed_experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            y = y.index_add(0, idx, self.experts[i](x[idx], weights[idx, top, None]).float())
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)


# ------------------------------------------------------------------------------------------ hyper-connections
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Torch port of the tilelang kernel: pre (sigmoid+eps), post (2*sigmoid), comb (Sinkhorn)."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Block(nn.Module):
    def __init__(self, layer_id, args: ModelArgs, shared, is_backbone=True):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args, shared, is_backbone)
        self.ffn = MoE(args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = nn.Parameter(torch.randn(mix_hc, hc * args.dim) * 0.01)
        self.hc_ffn_fn = nn.Parameter(torch.randn(mix_hc, hc * args.dim) * 0.01)
        base = torch.zeros(mix_hc)
        base[2 * hc:] = torch.eye(hc).flatten() * 4.0      # start near identity residual mixing
        self.hc_attn_base = nn.Parameter(base.clone())
        self.hc_ffn_base = nn.Parameter(base.clone())
        self.hc_attn_scale = nn.Parameter(torch.ones(3))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x, pre_mix, decode=None):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn_norm(self.hc_pre(x, pre_mix))
        h = self.attn(h) if decode is None else self.attn.decode(h, *decode)
        x = self.hc_post(h, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


def make_identity_pre_mix(x, hc_mult):
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


# ------------------------------------------------------------------------------------------ model
class Transformer(nn.Module):
    """embed (+ observation projections) -> hc copies -> CSA2 blocks -> collapse -> action logits / value / MTP."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.shared = SharedAttentionRuntime()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        # observation "patch" projections, one per observation token type (like the VL aligner)
        self.obs_proj = nn.ModuleList([nn.Sequential(nn.Linear(w, args.dim), nn.GELU(), nn.Linear(args.dim, args.dim))
                                       for w in args.obs_types])
        self.layers = nn.ModuleList([Block(i, args, self.shared) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.value_head = nn.Sequential(nn.Linear(args.dim, args.dim), nn.GELU(), nn.Linear(args.dim, args.n_value))
        # MTP (DeepSeek-V3 style): h_i and emb(t_{i+1}) -> block -> predicts t_{i+2}
        self.mtp = nn.ModuleList()
        for j in range(args.n_mtp_layers):
            m = nn.Module()
            m.hnorm, m.enorm = RMSNorm(args.dim), RMSNorm(args.dim)
            m.proj = nn.Linear(2 * args.dim, args.dim, bias=False)
            m.block = Block(args.n_layers + j, args, self.shared, is_backbone=False)
            m.norm = RMSNorm(args.dim)
            self.mtp.append(m)
        self.hc_mult = args.hc_mult

    def embed_inputs(self, ids, obs_feats, obs_type):
        """ids [b,s]; obs_feats list per type of [n_type, width] rows; obs_type [b,s] (-1 = action token);
        obs rows are consumed in sequence order per type."""
        h = self.embed(ids)
        if obs_feats is not None:
            for t, proj in enumerate(self.obs_proj):
                m = obs_type == t
                if m.any():
                    h = h.masked_scatter(m.unsqueeze(-1), (h[m] + proj(obs_feats[t].to(h.dtype))).to(h.dtype))
        return h

    def backbone(self, h, checkpoint=False):
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        aux = h.new_zeros((), dtype=torch.float32)

        sh = self.shared

        def run(layer, h, pre_mix, snap):
            # restore the cross-layer shared KV / indices this layer saw on the first pass (needed for
            # correct recomputation under activation checkpointing)
            sh.compress_kv, sh.index_k, sh.topk_idxs, sh.index_scores = snap
            h, pm = layer(h, pre_mix)
            a = layer.attn.aux_loss
            return h, pm, (a if a is not None else h.new_zeros((), dtype=torch.float32))

        for layer in self.layers:
            snap = (sh.compress_kv, sh.index_k, sh.topk_idxs, sh.index_scores)
            if checkpoint and self.training:
                h, pre_mix, a = torch.utils.checkpoint.checkpoint(run, layer, h, pre_mix, snap, use_reentrant=False)
            else:
                h, pre_mix, a = run(layer, h, pre_mix, snap)
            aux = aux + a
        self.last_aux = aux
        return self.norm(self.layers[-1].hc_pre(h, pre_mix))

    def forward_train(self, ids, obs_feats, obs_type, tgt_pos, tgt_ids, val_pos, mtp_pos=None, mtp_next=None,
                      mtp_tgt=None):
        """Returns (action logits at tgt_pos [N,V], value at val_pos [M,2], mtp logits, index aux loss).
        tgt_pos: [N,2] (batch, position) whose next token tgt_ids must be predicted."""
        h0 = self.embed_inputs(ids, obs_feats, obs_type)
        sh = self.shared
        sh.compress_kv = sh.index_k = sh.topk_idxs = sh.index_scores = None
        h = self.backbone(h0, checkpoint=True)
        logits = self.head(h[tgt_pos[:, 0], tgt_pos[:, 1]].float())
        value = self.value_head(h[val_pos[:, 0], val_pos[:, 1]].float())
        aux = self.last_aux
        mtp_logits = None
        if self.mtp and mtp_pos is not None:
            m = self.mtp[0]
            e = self.embed(torch.roll(ids, -1, 1))            # embedding of the next token
            x = m.proj(torch.cat([m.hnorm(h), m.enorm(e)], -1))
            x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
            x, pm = m.block(x, make_identity_pre_mix(x, self.hc_mult))
            x = m.norm(m.block.hc_pre(x, pm))
            mtp_logits = self.head(x[mtp_pos[:, 0], mtp_pos[:, 1]].float())
        return logits, value, mtp_logits, aux

    # ---------------------------------------------------------------------- incremental inference
    def init_cache(self, bsz=1, device="cpu"):
        return [layer.attn.init_cache(bsz, device) for layer in self.layers]

    @torch.no_grad()
    def step(self, h_in, start_pos, caches):
        """Feed ONE embedded token (h_in [b,1,dim]) at start_pos; returns final hidden [b,1,dim]."""
        h = h_in.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer, c in zip(self.layers, caches):
            h, pre_mix = layer(h, pre_mix, decode=(start_pos, c))
        return self.norm(self.layers[-1].hc_pre(h, pre_mix))

    def update_gate_bias(self, speed=1e-3):
        """noaux_tc: push the selection bias toward balanced expert load (outside the optimiser)."""
        for blk in list(self.layers) + [m.block for m in self.mtp]:
            load = blk.ffn.last_load
            if load is None:
                continue
            load = load.float()
            blk.ffn.gate.bias += speed * torch.sign(load.mean() - load)
