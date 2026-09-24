"""Consistency test: full-sequence (training) forward == incremental decode with caches.

Uses a real replay trajectory truncated to a few steps (so compressed-KV + indexer paths are exercised).
"""
import sys
import time

import numpy as np
import torch

from agent.obs_tokens import OBS_TYPES
from data.seq_extract import load_trajs
from model.dsv41 import ModelArgs, Transformer
from model.seq_batch import collate, obs_feats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/seq_test.npz"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    torch.manual_seed(0)
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 4096     # >= n_comp: selection = all visible (exact test)
    args = ModelArgs(obs_types=OBS_TYPES, max_seq_len=4096, window_size=32, index_topk=topk, attn_chunk=64)
    net = Transformer(args).eval()
    tr = load_trajs(path)[0]
    b = collate([tr], max_steps=steps)
    feats = obs_feats(b, torch.float32)
    t0 = time.time()
    with torch.no_grad():
        h0 = net.embed_inputs(b["ids"], feats, b["otype"])
        full = net.backbone(h0)
    t1 = time.time()
    caches = net.init_cache(1)
    outs = []
    with torch.no_grad():
        for p in range(h0.size(1)):
            outs.append(net.step(h0[:, p:p + 1], p, caches))
    inc = torch.cat(outs, 1)
    t2 = time.time()
    d = (full - inc).abs().max().item()
    print(f"S={h0.size(1)} max|full-incremental|={d:.2e} full {t1 - t0:.2f}s incremental {t2 - t1:.2f}s "
          f"({(t2 - t1) / h0.size(1) * 1e3:.2f} ms/token)")
    net.train()
    lg, v, mtp, aux = net.forward_train(b["ids"], feats, b["otype"], b["tgt_pos"], b["tgt_ids"], b["val_pos"],
                                        b["mtp_pos"], None, b["mtp_ids"])
    loss = torch.nn.functional.cross_entropy(lg, b["tgt_ids"]) + aux
    loss.backward()
    print("train loss", float(loss), "aux", float(aux), "logits", tuple(lg.shape), "mtp", tuple(mtp.shape),
          "params", sum(p.numel() for p in net.parameters()))
    assert d < 1e-3, d


if __name__ == "__main__":
    main()
