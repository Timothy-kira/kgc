"""TTT + Transformer market policy (PyTorch, for training).

Per step t:
    tokens   = 9 product tokens (PF feats + learned product embedding) + 1 global token (GF feats)
    Encoder  : 2-layer pre-LN Transformer over the 10 tokens (spatial / cross-product reasoning)
    z_t      = global-token output
Across steps (the long 720-step horizon):
    SWA      : causal sliding-window attention over z_{t-W+1..t}, W = 24 (one in-game day)
    TTT      : u_t = gelu(W1 h_t);  h'_t = h_t + W_fast(d) u_t
               W_fast is a fast weight updated at the end of every day d with one closed-form
               gradient step on a self-supervised loss that predicts that day's per-product
               market flow and price change (labels observable in-game):
                   L_d = || P_y W_fast ubar_d - y_d ||^2 ,  W_fast <- W_fast - eta * dL_d/dW_fast
               W_fast(0) and eta are meta-learned (TTT-E2E style, first-order through the unrolled update).
Heads:
    sell logits [9, N_BINS] from (product token output, h'_t); value from h'_t.
The numpy twin in model/policy_np.py reproduces this forward pass exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.features import GF, N_BINS, N_TGT, PF, BASE_BIN

D = 64
HEADS = 4
WIN = 24
N_PROD = 9


class Block(nn.Module):
    def __init__(self, d=D, heads=HEADS, mlp=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)
        self.heads = heads

    def attn(self, x, mask=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        y = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d)
        return self.proj(y)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TTTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pin = nn.Linear(PF, D)
        self.pemb = nn.Parameter(torch.zeros(N_PROD, D))
        self.gin = nn.Linear(GF, D)
        self.enc = nn.ModuleList([Block(), Block()])
        self.swa = Block()
        self.ln_t = nn.LayerNorm(D)
        self.w1 = nn.Linear(D, D)
        self.w_fast0 = nn.Parameter(torch.zeros(D, D))
        self.p_y = nn.Linear(D, N_TGT, bias=False)
        self.log_eta = nn.Parameter(torch.tensor(-3.0))
        # top-player prior head (pretrained by behaviour cloning on ladder replays, bins 0..4)
        self.thead = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, N_BINS - 1))
        # policy head sees product/temporal features + the top-player prior distribution
        self.head = nn.Sequential(nn.Linear(2 * D + N_BINS - 1, D), nn.GELU(), nn.Linear(D, N_BINS))
        self.vhead = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))  # [win logit, money diff/1e4]
        nn.init.normal_(self.pemb, std=0.02)
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[BASE_BIN] = 6.0   # start close to "follow the base executor"

    # ---- per-step spatial encoder
    def encode(self, P, G):
        """P [B,T,9,PF], G [B,T,GF] -> product outs [B,T,9,D], z [B,T,D]"""
        B, T = P.shape[:2]
        x = torch.cat([self.pin(P) + self.pemb, self.gin(G).unsqueeze(2)], dim=2)  # [B,T,10,D]
        x = x.view(B * T, N_PROD + 1, D)
        for blk in self.enc:
            x = blk(x)
        x = x.view(B, T, N_PROD + 1, D)
        return x[:, :, :N_PROD], x[:, :, N_PROD]

    def eta(self):
        return torch.sigmoid(self.log_eta) * 0.5

    # ---- temporal: SWA + chunked TTT over a whole episode
    def temporal(self, z, day_idx, day_tgt, day_mask):
        """z [B,T,D]; day_idx [T] (day of each step); day_tgt [B,ND,N_TGT]; day_mask [B,ND] (target valid).
        Returns h' [B,T,D] and the TTT self-supervised loss."""
        B, T, _ = z.shape
        ar = torch.arange(T, device=z.device)
        m = (ar[None, :] <= ar[:, None]) & (ar[None, :] > ar[:, None] - WIN)
        h = self.swa(z, m[None, None])
        u = F.gelu(self.w1(self.ln_t(h)))
        W = self.w_fast0.unsqueeze(0).expand(B, D, D)
        eta = self.eta()
        outs, ssl = [], z.new_zeros(())
        n_days = int(day_idx.max().item()) + 1
        for d in range(n_days):
            sl = (day_idx == d).nonzero().squeeze(1)
            ud = u[:, sl]                                          # [B,Td,D]
            outs.append(h[:, sl] + torch.einsum("bij,btj->bti", W, ud))
            if d < day_tgt.shape[1]:
                ub = ud.mean(1)                                    # [B,D]
                pred = self.p_y(torch.einsum("bij,bj->bi", W, ub))  # [B,N_TGT]
                err = (pred - day_tgt[:, d]) * day_mask[:, d:d + 1]
                ssl = ssl + (err ** 2).mean()
                # closed-form gradient of 0.5||P W ub - y||^2 wrt W:  (P^T err) ub^T
                g = torch.einsum("bi,bj->bij", err @ self.p_y.weight, ub)
                W = W - eta * g
        return torch.cat(outs, 1), ssl / max(1, n_days)

    def forward(self, P, G, day_idx, day_tgt, day_mask):
        po, z = self.encode(P, G)
        h, ssl = self.temporal(z, day_idx, day_tgt, day_mask)
        hx = h.unsqueeze(2).expand(-1, -1, N_PROD, -1)
        feat = torch.cat([po, hx], -1)
        tlogits = self.thead(feat)                                 # [B,T,9,N_BINS-1]
        logits = self.head(torch.cat([feat, tlogits.softmax(-1)], -1))  # [B,T,9,N_BINS]
        value = self.vhead(h)                                      # [B,T,2]
        self.last_tlogits = tlogits
        return logits, value, ssl

    def export(self, path):
        import numpy as np
        np.savez(path, **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})
