"""decode_chunk_multi (per-element chunked incremental decode) == token-by-token step_multi.

python -m tools.test_chunk_multi <ckpt_dir>   (model_args.json + engram.pt)
Games sit at different positions (prefixes of 0..300 tokens, crossing window wraps and compressor groups); a
35-token chunk is fed both ways, then two more single tokens to check the caches.
"""
import sys

import torch

from rl.tf_rl import load_net


def clone(c):
    if torch.is_tensor(c):
        return c.clone()
    if isinstance(c, dict):
        return {k: clone(v) for k, v in c.items()}
    if isinstance(c, list):
        return [clone(v) for v in c]
    return c


def main(ck):
    torch.manual_seed(0)
    net, _ = load_net(ck, None, "cpu")
    net.eval()
    prefix = [0, 1, 37, 127, 128, 129, 300]
    per = []
    with torch.no_grad():
        for L in prefix:
            c = net.init_cache(1, "cpu")
            for t in range(L):
                net.step(torch.randn(1, 1, net.args.dim), t, c)
            per.append(c)
        caches = net.stack_caches(per)
        B = len(prefix)
        pos = torch.tensor(prefix)
        rows_s = torch.randint(0, 1000, (B, 1, net.args.engram_n_hash_cols))
        rows_d = torch.randint(0, 1000, (B, 1, net.args.engram_n_hash_cols))
        net.engram_ids, net.engram_mask = {1: rows_s, 3: rows_d}, None
        x = torch.randn(B, 35, net.args.dim)
        ca, cb = clone(caches), clone(caches)
        ha = torch.cat([net.step_multi(x[:, i:i + 1], pos + i, ca) for i in range(35)], 1)
        hb = net.step_multi(x, pos, cb)
        e1 = (ha - hb).abs().max().item()
        y = torch.randn(B, 2, net.args.dim)
        ya = torch.cat([net.step_multi(y[:, i:i + 1], pos + 35 + i, ca) for i in range(2)], 1)
        yb = torch.cat([net.step_multi(y[:, i:i + 1], pos + 35 + i, cb) for i in range(2)], 1)
        e2 = (ya - yb).abs().max().item()
    print("chunk max err", e1, "after-cache max err", e2, "scale", ha.abs().mean().item())
    assert e1 < 1e-3 and e2 < 1e-3
    print("CHUNK_MULTI_OK")


if __name__ == "__main__":
    main(sys.argv[1])
