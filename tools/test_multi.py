"""Batched per-element-position decode (step_multi) == per-game decode."""
import torch

from agent.obs_tokens import OBS_TYPES
from data.seq_extract import load_trajs
from model.dsv41 import ModelArgs, Transformer
from model.seq_batch import collate, obs_feats

torch.manual_seed(0)
net = Transformer(ModelArgs(obs_types=OBS_TYPES, max_seq_len=4096, window_size=32, index_topk=8)).eval()
tr = load_trajs("/tmp/seq_test.npz")
hs = []
for k in range(2):
    b = collate([tr[k]], max_steps=6)
    with torch.no_grad():
        hs.append(net.embed_inputs(b["ids"], obs_feats(b, torch.float32), b["otype"]))
# references: each sequence decoded alone
refs = []
for h in hs:
    c = net.init_cache(1)
    with torch.no_grad():
        refs.append(torch.cat([net.step(h[:, p:p + 1], p, c) for p in range(h.size(1))], 1))
# element 0 starts at 0, element 1 is first advanced alone by `off` tokens, then both run batched
off = 37
c0, c1 = net.init_cache(1), net.init_cache(1)
with torch.no_grad():
    for p in range(off):
        net.step(hs[1][:, p:p + 1], p, c1)
caches = Transformer.stack_caches([c0, c1])
n = min(hs[0].size(1), hs[1].size(1) - off)
out0, out1 = [], []
with torch.no_grad():
    for t in range(n):
        x = torch.cat([hs[0][:, t:t + 1], hs[1][:, off + t:off + t + 1]], 0)
        o = net.step_multi(x, torch.tensor([t, off + t]), caches)
        out0.append(o[0:1]); out1.append(o[1:2])
d0 = (torch.cat(out0, 1) - refs[0][:, :n]).abs().max().item()
d1 = (torch.cat(out1, 1) - refs[1][:, off:off + n]).abs().max().item()
print(f"n={n} max diff elem0={d0:.2e} elem1={d1:.2e}")
assert d0 < 1e-4 and d1 < 1e-4
