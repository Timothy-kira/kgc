"""Train the H5 Engram route memory (model/route_engram.py) on route-value data (tools/route_data.py).

python -m tools.train_route <out.pt> <route_*.jsonl,...> [epochs=60] [margin=0.1] [holdout=0.2]
Target per game and route: (money diff of that route - money diff of cha22's own route) / 10000, where money diff =
our money - opponent money (the win objective). Every route of a game is observed (fork data), so the policy
"switch to argmax when the predicted gain exceeds `margin`" is evaluated EXACTLY on held-out games: mean realised
gain vs cha22's route, win-rate change, switch rate. Holdout is split by seed (whole games).
"""
import glob
import json
import random
import sys

import numpy as np
import torch

from model.route_engram import RouteEngram, RouteVocab


def load(patterns):
    rows = []
    for pat in patterns.split(","):
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if r.get("default_route") is not None and str(r["default_route"]) in r["results"]:
                        rows.append(r)
    return rows


def tensors(rows, vocab, routes):
    X, Y, M, D = [], [], [], []
    for r in rows:
        X.append(vocab.sequence(r["d_codes"], r["shops2"]))
        d0 = r["results"][str(r["default_route"])]
        base = d0[0] - d0[1]
        y, m = np.zeros(len(routes), np.float32), np.zeros(len(routes), np.float32)
        for k, rt in enumerate(routes):
            v = r["results"].get(str(rt))
            if v is not None:
                y[k], m[k] = ((v[0] - v[1]) - base) / 10000.0, 1.0
        Y.append(y), M.append(m), D.append(routes.index(r["default_route"]))
    return np.stack(X), torch.tensor(np.stack(Y)), torch.tensor(np.stack(M)), np.array(D)


def evaluate(model, X, Y, M, D, routes, rows, margin):
    with torch.no_grad():
        pred = model(X).numpy()
    gains, switches, win0, win1 = [], 0, 0, 0
    for i, r in enumerate(rows):
        p = np.where(M[i].numpy() > 0, pred[i], -1e9)
        k = int(p.argmax())
        pick = k if p[k] - pred[i][D[i]] > margin else D[i]
        switches += pick != D[i]
        g = float(Y[i][pick]) * 10000
        gains.append(g)
        d0 = r["results"][str(r["default_route"])]
        win0 += d0[0] > d0[1]
        win1 += (d0[0] - d0[1] + g) > 0
    n = max(1, len(rows))
    oracle = float(np.mean([float(Y[i].max()) * 10000 for i in range(len(rows))])) if rows else 0.0
    return {"n": len(rows), "mean_gain": round(float(np.mean(gains)) if gains else 0.0), "oracle_gain": round(oracle),
            "switch_rate": round(switches / n, 3), "win_cha22": round(win0 / n, 3), "win_model": round(win1 / n, 3)}


def main():
    out, pats = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    hold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    rows = load(pats)
    routes = sorted({int(k) for r in rows for k in r["results"]})
    seeds = sorted({(r["seed"], r["opponent"]) for r in rows})
    random.Random(0).shuffle(seeds)
    test_keys = set(seeds[:int(len(seeds) * hold)])
    tr = [r for r in rows if (r["seed"], r["opponent"]) not in test_keys]
    te = [r for r in rows if (r["seed"], r["opponent"]) in test_keys]
    vocab = RouteVocab(codes=[c for r in tr for c in r["d_codes"][:7]], min_count=1)
    torch.manual_seed(0)
    model = RouteEngram(vocab.size, routes)
    Xtr, Ytr, Mtr, Dtr = tensors(tr, vocab, routes)
    Xte, Yte, Mte, Dte = tensors(te, vocab, routes) if te else (None,) * 4
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    print(f"games train {len(tr)} test {len(te)} routes {len(routes)} vocab {vocab.size}", flush=True)
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        for s in range(0, len(perm), 64):
            idx = perm[s:s + 64]
            pred = model(Xtr[idx])
            loss = (((pred - Ytr[idx]) ** 2) * Mtr[idx]).sum() / Mtr[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            model.eval()
            msg = {"epoch": ep + 1, "loss": round(float(loss), 4),
                   "train": evaluate(model, Xtr, Ytr, Mtr, Dtr, routes, tr, margin)}
            if te:
                msg["test"] = evaluate(model, Xte, Yte, Mte, Dte, routes, te, margin)
            print(json.dumps(msg), flush=True)
    torch.save({"state": model.state_dict(), "routes": routes, "vocab_keys": vocab.keys, "margin": margin}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
