"""Stage 1 (pre-train) and stage 2 (mid-train) for the TTT+Transformer market policy.

Pre-train  (--stage pre): all ladder replays (rating >= 1800, both seats), streamed shard by shard.
    * top-player prior head: behaviour cloning of every player's sell decisions, weighted by
      rating and outcome (winner 1.0 / loser 0.5)
    * value head: final win (BCE) + money diff (MSE) from every step
    * TTT self-supervised loss: each day's per-product market flow / price change
Mid-train  (--stage mid, --init pre.pt): high-quality subset (rating >= --min_score, default 2800)
    at lower LR, mixed with traces of our own base executor (rl/collect_base.py) so the policy head
    is aligned with the executor: it learns to output BASE (= exactly the executor) while the
    prior head keeps the top-player knowledge; TTT is meta-learned across the many opponents.
Post-train is RL (rl/grpo.py --init mid.pt).

python -m model.pretrain --stage pre --data <ext_dir> --out pre.pt [--epochs 2]
python -m model.pretrain --stage mid --data <ext_dir> --base_traces <dir> --init pre.pt --out mid.pt
"""
import argparse
import glob
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.features import BASE_BIN
from model.net import TTTPolicy

KEYS = ("P", "G", "y", "day_tgt", "day_mask", "win", "diff", "score")


def iter_items(files, min_score, buffer_parts=4, seed=0):
    """Stream trajectories from npz parts; shuffle within a rolling buffer of parts."""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    for i in range(0, len(files), buffer_parts):
        buf = []
        for f in files[i:i + buffer_parts]:
            try:
                z = np.load(f)
                n = int(z["n"])
            except Exception:
                continue
            for j in range(n):
                it = {k: z[f"{k}_{j}"] for k in KEYS if f"{k}_{j}" in z}
                if float(it.get("score", 0)) >= min_score:
                    buf.append(it)
        rng.shuffle(buf)
        yield from buf


def batch(items):
    T = max(len(it["P"]) for it in items)
    B = len(items)
    nd = (T + 23) // 24
    P = np.zeros((B, T) + items[0]["P"].shape[1:], np.float32)
    G = np.zeros((B, T, items[0]["G"].shape[-1]), np.float32)
    y = np.zeros((B, T, 9), np.int64)
    valid = np.zeros((B, T), bool)
    dt = np.zeros((B, nd, items[0]["day_tgt"].shape[-1]), np.float32)
    dm = np.zeros((B, nd), np.float32)
    is_base = np.zeros(B, bool)
    for i, it in enumerate(items):
        n = len(it["P"])
        P[i, :n], G[i, :n], y[i, :n], valid[i, :n] = it["P"], it["G"], it["y"], True
        k = min(nd, len(it["day_mask"]))
        dt[i, :k], dm[i, :k] = it["day_tgt"][:k], it["day_mask"][:k]
        is_base[i] = bool(it.get("is_base", False))
    w = np.array([(1.0 if it["win"] > 0.5 else 0.5) * min(1.5, max(0.3, (float(it["score"]) - 1500) / 1500))
                  for it in items], np.float32)
    win = np.array([it["win"] for it in items], np.float32)
    diff = np.array([it["diff"] for it in items], np.float32)
    out = [torch.tensor(x) for x in (P, G, y, valid, dt, dm, w, win, diff, is_base)]
    return out + [torch.arange(T) // 24]


def loss_fn(net, bt, policy_coef):
    P, G, y, valid, dt, dm, w, win, diff, is_base, day_idx = bt
    logits, value, ssl = net(P, G, day_idx, dt, dm)
    tl = net.last_tlogits
    has = (P[..., 2] > 0) & valid.unsqueeze(-1)
    top = has & ~is_base.view(-1, 1, 1)
    ce = F.cross_entropy(tl.reshape(-1, tl.shape[-1]), y.clamp(max=tl.shape[-1] - 1).reshape(-1),
                         reduction="none").view(y.shape)
    wt = w * (~is_base).float()
    ce = ((ce * top).sum((1, 2)) / top.sum((1, 2)).clamp(min=1) * wt).sum() / wt.sum().clamp(min=1e-6)
    acc = ((tl.argmax(-1) == y) & top).sum() / top.sum().clamp(min=1)
    nz = top & (y > 0)
    acc_sell = ((tl.argmax(-1) == y) & nz).sum() / nz.sum().clamp(min=1)
    vw = F.binary_cross_entropy_with_logits(value[..., 0], win.view(-1, 1).expand_as(value[..., 0]), reduction="none")
    vd = (value[..., 1] - diff.view(-1, 1)) ** 2
    vloss = ((vw + 0.1 * vd) * valid).sum() / valid.sum()
    loss = ce + 0.3 * vloss + 0.2 * ssl
    st = dict(ce=float(ce), acc=float(acc), acc_sell=float(acc_sell), vloss=float(vloss), ssl=float(ssl))
    if policy_coef > 0 and bool(is_base.any()):
        # mid-train: align the policy head with our executor (label BASE) on executor traces
        bm = has & is_base.view(-1, 1, 1)
        lp = torch.log_softmax(logits, -1)[..., BASE_BIN]
        pol = -(lp * bm).sum() / bm.sum().clamp(min=1)
        loss = loss + policy_coef * pol
        st["pol_base"] = float(pol)
    return loss, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "mid"], default="pre")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_traces", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min_score", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val_parts", type=int, default=2)
    ap.add_argument("--max_hours", type=float, default=100)
    a = ap.parse_args()
    ms = a.min_score if a.min_score is not None else (1800 if a.stage == "pre" else 2800)
    lr = a.lr or (1e-3 if a.stage == "pre" else 3e-4)
    files = sorted(glob.glob(a.data + "/part_*.npz"))
    val_files, train_files = files[:a.val_parts], files[a.val_parts:]
    base_files = sorted(glob.glob(a.base_traces + "/*.npz")) if a.base_traces else []
    print(f"stage={a.stage} parts train={len(train_files)} val={len(val_files)} base={len(base_files)} "
          f"min_score={ms}", flush=True)
    net = TTTPolicy()
    if a.init:
        print("init", net.load_state_dict(torch.load(a.init), strict=False), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    deadline = time.time() + a.max_hours * 3600
    t0 = time.time()
    step = 0
    policy_coef = 1.0 if a.stage == "mid" else 0.0
    for ep in range(a.epochs):
        lr_ep = lr * (0.5 ** ep)
        for g in opt.param_groups:
            g["lr"] = lr_ep
        net.train()
        base_iter = iter_items(base_files, -1e9, seed=ep) if base_files else None
        buf = []
        for it in iter_items(train_files, ms, seed=ep):
            buf.append(it)
            if base_iter is not None and random.random() < 0.3:   # mid-train mixture: 30% executor traces
                b = next(base_iter, None)
                if b is not None:
                    b["is_base"] = True
                    buf.append(b)
            if len(buf) < a.batch:
                continue
            loss, st = loss_fn(net, batch(buf[:a.batch]), policy_coef)
            buf = buf[a.batch:]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step} {({k: round(v, 4) for k, v in st.items()})} {time.time() - t0:.0f}s",
                      flush=True)
            if step % 500 == 0:
                torch.save(net.state_dict(), a.out)
            if time.time() > deadline:
                break
        net.eval()
        with torch.no_grad():
            vs, vb = [], []
            for it in iter_items(val_files, ms, seed=123):
                vb.append(it)
                if len(vb) == a.batch:
                    vs.append(loss_fn(net, batch(vb), 0.0)[1])
                    vb = []
                if len(vs) >= 20:
                    break
        if vs:
            print("VAL", ep, {k: round(float(np.mean([v[k] for v in vs])), 4) for k in vs[0]}, flush=True)
        torch.save(net.state_dict(), a.out)
        if time.time() > deadline:
            break
    net.export(a.out.replace(".pt", ".npz"))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
