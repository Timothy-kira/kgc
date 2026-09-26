"""v5 residual policy (docs/PLAN_RL.md, section 2): a small Transformer over the last WINDOW decision tokens with
official Engram layers (model/engram.py, hc_mult = 1) on the opponent's S / D event streams before the blocks.
Output: 5 heads x 3 actions (follow / hold / dump); the follow logit starts at +FOLLOW_BIAS so the initial greedy
policy is exactly v4b."""
import numpy as np
import torch
import torch.nn as nn

from model.engram import Engram
from rl.v5_agent import CTRL, FEAT_DIM, FOLLOW, N_ACT, WINDOW, engram_layout

FOLLOW_BIAS = 2.0


class RLPolicy(nn.Module):
    def __init__(self, vocab_sizes, dim=128, n_layers=2, n_heads=4, follow_bias=FOLLOW_BIAS):
        super().__init__()
        lay = engram_layout(vocab_sizes)
        self.inp = nn.Sequential(nn.Linear(FEAT_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(WINDOW, dim))
        self.eng_s = Engram(dim, 1, lay.rows(0), lay.n_hash_cols, lay.head_dim)
        self.eng_d = Engram(dim, 1, lay.rows(1), lay.n_hash_cols, lay.head_dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(CTRL) * N_ACT)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()
            self.head.bias.view(len(CTRL), N_ACT)[:, FOLLOW] = follow_bias

    def forward(self, feats, rows_s, rows_d, mask, allowed):
        """feats [B,W,F], rows_* [B,W,C] int64, mask [B,W] bool (valid tokens; the last is the current decision),
        allowed [B,5,3] bool -> masked logits [B,5,3]."""
        x = self.inp(feats) + self.pos
        x = self.eng_s(x.unsqueeze(2), rows_s, mask).squeeze(2)
        x = self.eng_d(x.unsqueeze(2), rows_d, mask).squeeze(2)
        h = self.blocks(x, src_key_padding_mask=~mask)
        lg = self.head(self.norm(h[:, -1])).view(-1, len(CTRL), N_ACT)
        return lg.masked_fill(~allowed, -1e9)

    def param_groups(self, lr, table_mult=5.0):
        tab = [self.eng_s.embed.weight, self.eng_d.embed.weight]
        ids = {id(p) for p in tab}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": rest, "lr": lr, "weight_decay": 0.01},
                {"params": tab, "lr": lr * table_mult, "weight_decay": 0.0}]    # Engram paper: 5x lr, no wd


def batch_inputs(inps, device="cpu"):
    """list of V5Agent.inputs dicts -> tensors."""
    t = lambda k, dt: torch.from_numpy(np.stack([i[k] for i in inps])).to(device=device, dtype=dt)
    return (t("feats", torch.float32), t("rows_s", torch.long), t("rows_d", torch.long), t("mask", torch.bool),
            t("allowed", torch.bool))


class PolicyRunner:
    """CPU inference for actors: greedy joint action and per-head probabilities."""

    def __init__(self, net):
        self.net = net.eval()

    @torch.no_grad()
    def logits(self, inp):
        return self.net(*batch_inputs([inp]))[0]

    def greedy(self, inp):
        lg = self.logits(inp)
        return [int(a) for a in lg.argmax(-1)], None

    def probs(self, inp):
        return torch.softmax(self.logits(inp), -1).numpy()
