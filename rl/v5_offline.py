"""Pooled offline learner for the A track (docs/PLAN_RL.md, section 10): one-step policy improvement over v4b.

python -m rl.v5_offline --samples '<glob of samples_*.pkl>' --vocab opp_vocab.npz --out DIR
       [--eng_init engram.pt] [--freeze_tables] [--ens 5] [--epochs 30] [--hold 0.1]

Samples come from `rl.v5_rl --sample_only` kernels (v4b acting; every alternative action of one head enumerated and
rolled out exactly with the v4b continuation), so all sampler kernels pool into one dataset of exact Q^{v4b}(s, a)
differences. The bootstrap ensemble advantage model (same loss as rl/v5_rl.update --obj adv) is fit on 90% of the
games; the held-out 10% (split by game, never by sample) gives the true single-decision value of acting with
"deviate from follow iff mean - c * std of the predicted gain > margin" for a grid of (margin, c), which picks the
acting rule. --eng_init copies the pretrained CSA2 Engram tables (model/train_clm.py --stage engram; same layout
since rl/v5_agent.BUCKET_START = 2**15) into the policy's Engram layers.
Output: DIR/policy.pt (weights), DIR/policy.json (margin, c, ens, holdout stats), DIR/grid.json.
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.opp_data import Vocab
from rl.v5_agent import FOLLOW, N_ACT
from rl.v5_policy import RLPolicy

KEYS = ("feats", "rows_s", "rows_d", "mask", "allowed")
DTYPES = (torch.float32, torch.long, torch.long, torch.bool, torch.bool)


def load(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    S = []
    for f in files:
        try:
            S += pickle.load(open(f, "rb"))
        except Exception as e:                                   # a truncated file must not stop the learner
            print("skip", f, e, flush=True)
    print(json.dumps({"files": len(files), "samples": len(S)}), flush=True)
    return S


def tensors(S, idx):
    X = [torch.from_numpy(np.stack([S[i]["inp"][k] for i in idx])) for k in KEYS]
    X = [x.to(dt) for x, dt in zip(X, DTYPES)]
    R = torch.from_numpy(np.stack([S[i]["r"] for i in idx]).astype(np.float32))
    k = torch.tensor([S[i]["head"] for i in idx])
    g = torch.tensor([S[i]["greedy"] for i in idx])
    return X, R, k, g


def eng_init(net, path, freeze):
    sd = torch.load(path, map_location="cpu")
    for mine, layer in ((net.eng_s, 1), (net.eng_d, 3)):
        pre = f"engrams.{layer}."
        emb = sd[pre + "embed.weight"]
        assert emb.shape == mine.embed.weight.shape, (emb.shape, mine.embed.weight.shape)
        with torch.no_grad():
            mine.embed.weight.copy_(emb)
            w, d = sd[pre + "wkv.weight"], mine.dim            # CSA2: [(hc+1)*d, C*hd], hc = 2; here hc = 1
            if w.shape[1] == mine.wkv.weight.shape[1] and w.shape[0] >= 2 * d:
                mine.wkv.weight[:d].copy_(w[:d])                # key of the first hyper-connection stream
                mine.wkv.weight[d:].copy_(w[-d:] * 0.1)         # value, damped: start close to the untransplanted net
            mine.q_weight.copy_(sd[pre + "q_weight"][:1])
            mine.k_weight.copy_(sd[pre + "k_weight"][:1])
        if freeze:
            mine.embed.weight.requires_grad_(False)
    print("eng_init", path, "freeze" if freeze else "", flush=True)


def adv_loss(net, xb, R, k, g, boot):
    raw = net.forward_all(*xb)                                    # [n,E,5,3]
    ii = torch.arange(len(k), device=R.device)
    E = raw.size(1)
    lgk = raw[ii, :, k]                                           # [n,E,3]
    pred = lgk - lgk.gather(2, g.view(-1, 1, 1).expand(-1, E, 1))
    valid = ~torch.isnan(R)
    tgt = ((R - R.gather(1, g[:, None])) / 0.01).clamp(-50, 50)
    tgt = torch.nan_to_num(tgt).unsqueeze(1).expand_as(pred)
    m = (valid & (torch.arange(N_ACT, device=R.device)[None] != g[:, None])).unsqueeze(1).float()
    w = boot.unsqueeze(-1)
    el = F.huber_loss(pred, tgt, delta=5.0, reduction="none")
    den = (m * w).sum().clamp_min(1.0)
    return (el * m * w).sum() / den


@torch.no_grad()
def gains(net, X, k, bs, dev):
    out = []
    for i in range(0, len(k), bs):
        xb = tuple(x[i:i + bs].to(dev) for x in X)
        la = net.forward_all(*xb)                                # [n,E,5,3]
        ii = torch.arange(la.size(0), device=dev)
        lk = la[ii, :, k[i:i + bs].to(dev)]                       # [n,E,3]
        out.append((lk - lk[..., FOLLOW:FOLLOW + 1]).cpu())       # predicted gain over follow, 0.01 reward units
    return torch.cat(out)


def decision_value(G, allowed_k, R, g, margin, c):
    """Mean realised reward gain (vs the greedy = v4b action) of the acting rule on held-out decisions."""
    score = G.mean(1) - (c * G.std(1) if G.size(1) > 1 else 0)
    score[:, FOLLOW] = margin
    score = score.masked_fill(~allowed_k, -1e9)
    a = score.argmax(-1)
    ra = R.gather(1, a[:, None]).squeeze(1)
    rg = R.gather(1, g[:, None]).squeeze(1)
    gain = torch.where(torch.isnan(ra), torch.zeros_like(ra), ra - rg)
    return float(gain.mean()), float((a != g).float().mean()), float(gain.std() / max(1, len(gain)) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eng_init", default="")
    ap.add_argument("--freeze_tables", action="store_true")
    ap.add_argument("--ens", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hold", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = load(a.samples)
    jobs = np.array([hash((s.get("job", i), 7)) % 1000 for i, s in enumerate(S)])
    hold = np.where(jobs < a.hold * 1000)[0]
    train = np.where(jobs >= a.hold * 1000)[0]
    Xt, Rt, kt, gt = tensors(S, train)
    Xh, Rh, kh, gh = tensors(S, hold)
    del S
    print(json.dumps({"train": len(train), "hold": len(hold), "dev": dev}), flush=True)
    vocab = Vocab(a.vocab)
    torch.manual_seed(0)
    net = RLPolicy(vocab.sizes, ens=a.ens)
    if a.eng_init:
        eng_init(net, a.eng_init, a.freeze_tables)
    net.to(dev)
    groups = [dict(g_, params=[p for p in g_["params"] if p.requires_grad]) for g_ in net.param_groups(a.lr)]
    opt = torch.optim.AdamW([g_ for g_ in groups if g_["params"]], betas=(0.9, 0.99))
    boot = torch.poisson(torch.ones(len(kt), a.ens))              # fixed bootstrap weights per member
    oracle = Rh.nan_to_num(-1e9).max(1).values - Rh.gather(1, gh[:, None]).squeeze(1)
    print(json.dumps({"hold_oracle_gain": round(float(oracle.mean()), 5),
                      "hold_better": round(float((oracle > 0).float().mean()), 3)}), flush=True)
    allowed_h = Xh[4][torch.arange(len(kh)), kh]
    best, bad, t0 = (-1e9, None), 0, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(kt))
        tl = []
        for i in range(0, len(kt), a.bs):
            idx = perm[i:i + a.bs]
            xb = tuple(x[idx].to(dev, non_blocking=True) for x in Xt)
            loss = adv_loss(net, xb, Rt[idx].to(dev), kt[idx].to(dev), gt[idx].to(dev), boot[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl.append(float(loss))
        net.eval()
        with torch.no_grad():
            hl = []
            for i in range(0, len(kh), a.bs):
                xb = tuple(x[i:i + a.bs].to(dev) for x in Xh)
                hl.append(float(adv_loss(net, xb, Rh[i:i + a.bs].to(dev), kh[i:i + a.bs].to(dev), gh[i:i + a.bs].to(dev),
                                         torch.ones(min(a.bs, len(kh) - i), a.ens, device=dev))))
        G = gains(net, Xh, kh, a.bs, dev)
        grid = {}
        for m in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
            for c in (0.0, 0.5, 1.0, 2.0):
                grid[f"{m}/{c}"] = decision_value(G, allowed_h, Rh, gh, m, c)
        key, (v, dv, se) = max(grid.items(), key=lambda kv: kv[1][0])
        rec = {"ep": ep, "train_loss": round(float(np.mean(tl)), 4), "hold_loss": round(float(np.mean(hl)), 4),
               "best_rule": key, "hold_gain": round(v, 5), "hold_gain_se": round(se, 5), "dev_rate": round(dv, 4),
               "min": round((time.time() - t0) / 60, 1)}
        print("EPOCH " + json.dumps(rec), flush=True)
        if v > best[0]:
            m, c = (float(x) for x in key.split("/"))
            best, bad = (v, key), 0
            torch.save({k_: v_.cpu() for k_, v_ in net.state_dict().items()}, os.path.join(a.out, "policy.pt"))
            json.dump({"margin": m, "c": c, "ens": a.ens, "eng_init": a.eng_init, **rec},
                      open(os.path.join(a.out, "policy.json"), "w"))
            json.dump({k_: list(v_) for k_, v_ in grid.items()}, open(os.path.join(a.out, "grid.json"), "w"))
        else:
            bad += 1
        if bad >= a.patience or time.time() - t0 > a.hours * 3600:
            break
    print("BEST " + json.dumps({"hold_gain": best[0], "rule": best[1]}), flush=True)


if __name__ == "__main__":
    main()
