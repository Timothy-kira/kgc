"""Pretrain the TTT+Transformer trunk on ladder replays (SFT cold start).

Losses (per episode-seat trajectory):
  - top-player prior head: CE on the player's actual sell bins (only where it had stock),
    weighted by trajectory quality (winner 1.0 / loser 0.5, x rating factor)  [CodeMidas-style quality filter]
  - value head: final win (BCE) + money diff (MSE) from every step
  - TTT self-supervised loss (market flow / price change of each day)
The policy head keeps its "follow the base executor" initialisation.

python -m model.pretrain <extract_dir> <out.pt> [epochs] [max_parts]
"""
import glob
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.net import TTTPolicy


def load_parts(d, max_parts=10 ** 9):
    items = []
    for f in sorted(glob.glob(d + "/part_*.npz"))[:max_parts]:
        z = np.load(f)
        for i in range(int(z["n"])):
            items.append({k: z[f"{k}_{i}"] for k in ("P", "G", "y", "day_tgt", "day_mask", "win", "diff", "score")})
    return items


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
    for i, it in enumerate(items):
        n = len(it["P"])
        P[i, :n], G[i, :n], y[i, :n], valid[i, :n] = it["P"], it["G"], it["y"], True
        k = min(nd, len(it["day_mask"]))
        dt[i, :k], dm[i, :k] = it["day_tgt"][:k], it["day_mask"][:k]
    w = np.array([(1.0 if it["win"] > 0.5 else 0.5) * min(1.5, max(0.3, (float(it["score"]) - 1500) / 1500))
                  for it in items], np.float32)
    win = np.array([it["win"] for it in items], np.float32)
    diff = np.array([it["diff"] for it in items], np.float32)
    return [torch.tensor(x) for x in (P, G, y, valid, dt, dm, w, win, diff)] + [torch.arange(T) // 24]


def main():
    src, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_parts = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    items = load_parts(src, max_parts)
    random.seed(0)
    random.shuffle(items)
    nval = max(8, len(items) // 20)
    val, train = items[:nval], items[nval:]
    print(f"train {len(train)} val {len(val)}", flush=True)
    net = TTTPolicy()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    B = 16
    steps = epochs * (len(train) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=max(1, steps))

    def loss_fn(bt):
        P, G, y, valid, dt, dm, w, win, diff, day_idx = bt
        _, value, ssl = net(P, G, day_idx, dt, dm)
        tl = net.last_tlogits
        has = (P[..., 2] > 0) & valid.unsqueeze(-1)
        ce = F.cross_entropy(tl.reshape(-1, tl.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
        ce = ((ce * has).sum((1, 2)) / has.sum((1, 2)).clamp(min=1) * w).sum() / w.sum()
        acc = (((tl.argmax(-1) == y) & has).sum() / has.sum().clamp(min=1))
        nz = has & (y > 0)
        acc_nz = (((tl.argmax(-1) == y) & nz).sum() / nz.sum().clamp(min=1))
        vw = F.binary_cross_entropy_with_logits(value[..., 0], win.view(-1, 1).expand_as(value[..., 0]), reduction="none")
        vd = (value[..., 1] - diff.view(-1, 1)) ** 2
        vloss = ((vw + 0.1 * vd) * valid).sum() / valid.sum()
        return ce + 0.3 * vloss + 0.2 * ssl, dict(ce=float(ce), acc=float(acc), acc_sell=float(acc_nz),
                                                   vloss=float(vloss), ssl=float(ssl))

    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(train)
        net.train()
        for s in range(0, len(train) - B + 1, B):
            loss, st = loss_fn(batch(train[s:s + B]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if (s // B) % 20 == 0:
                print(f"ep {ep} step {s // B} {st} {time.time() - t0:.0f}s", flush=True)
        net.eval()
        with torch.no_grad():
            vs = [loss_fn(batch(val[i:i + B]))[1] for i in range(0, len(val), B)]
        print("VAL", ep, {k: round(float(np.mean([v[k] for v in vs])), 4) for k in vs[0]}, flush=True)
        torch.save(net.state_dict(), out)
    net.export(out.replace(".pt", ".npz"))


if __name__ == "__main__":
    main()
