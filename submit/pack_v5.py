"""Pack the v5 agent (v4b = cha22 + clamp_sells, with the learned hold / dump residual of rl/v5_rl.py) into one
Kaggle main.py.

python -m submit.pack_v5 <out_main.py> <policy.pt> <opp_vocab.npz> [margin=0] [c=0]
Embedded: league/cha22.py (its own module), model/engram.py, agent/opp_events.py, model/opp_data.py, rl/v5_agent.py,
rl/v5_policy.py, the compressed opponent-event vocab and the policy weights (Engram tables int8 with a per-row fp16
scale; ensemble size read from the head shape; acting rule = rl/v5_policy.PolicyRunner(margin, c)). Any exception in
the residual path falls back to v4b's own action (rl/v5_agent.V5Agent.__call__).
"""
import base64
import io
import sys

import torch

ROOT = "/home/user/kgc/"
MODULES = {"model.engram": "model/engram.py", "agent.opp_events": "agent/opp_events.py",
           "model.opp_data": "model/opp_data.py", "agent.loader": "agent/loader.py", "rl.v5_agent": "rl/v5_agent.py",
           "rl.v5_policy": "rl/v5_policy.py", "cha22_base": "league/cha22.py"}

TEMPLATE = r'''# Kaggriculture v5: v4b (cha22 + clamp_sells) + learned hold/dump residual (Transformer + Engram policy, RL).
import base64 as _b64, sys as _sys, types as _types, io as _io
_SRC = __SRC__
_W = __W__
_VOCAB = __VOCAB__

def _load(name):
    m = _types.ModuleType(name)
    m.__dict__["__name__"] = name
    _sys.modules[name] = m
    pkg = name.rsplit(".", 1)[0] if "." in name else None
    exec(compile(_b64.b64decode(_SRC[name]).decode(), name + ".py", "exec"), m.__dict__)
    if pkg:
        setattr(_sys.modules[pkg], name.rsplit(".", 1)[1], m)
    return m

for _pkg in ("model", "agent", "rl"):
    if _pkg not in _sys.modules:
        _sys.modules[_pkg] = _types.ModuleType(_pkg)
for _n in ("model.engram", "agent.opp_events", "model.opp_data", "agent.loader", "rl.v5_agent", "rl.v5_policy"):
    _load(_n)
_base = _load("cha22_base")
_FALLBACK = None

def _make():
    import torch
    torch.set_num_threads(1)
    A, P = _sys.modules["rl.v5_agent"], _sys.modules["rl.v5_policy"]
    vocab = _sys.modules["model.opp_data"].Vocab(_io.BytesIO(_b64.b64decode(_VOCAB)))
    raw = torch.load(_io.BytesIO(_b64.b64decode(_W)), map_location="cpu")
    sd = {}
    for k, v in raw.items():
        if k.endswith(".q8"):
            sd[k[:-3]] = v.float() * raw[k[:-3] + ".s8"].float()[:, None]
        elif not k.endswith(".s8"):
            sd[k] = v.float()
    ens = sd["head.weight"].shape[0] // 15
    net = P.RLPolicy(vocab.sizes, ens=ens)
    net.load_state_dict(sd)
    net.eval()
    return A.V5Agent(vocab, policy_fn=P.PolicyRunner(net, __MARGIN__, __C__).greedy, module=_base)

_AGENT = None

def agent(obs, config=None):
    global _AGENT, _FALLBACK
    if _AGENT is None and _FALLBACK is None:
        try:
            _AGENT = _make()
        except Exception:
            loader = _sys.modules["agent.loader"]
            _base._IMPL.chassis.cfg.update({"clamp_sells": True})
            _FALLBACK = loader.call_adapter(getattr(_base, loader.entry_name(_base)))
    if _AGENT is not None:
        return _AGENT(obs, config)
    return _FALLBACK(obs, config)
'''


def quant(sd):
    out = {}
    for k, v in sd.items():
        if "embed.weight" in k:
            s = v.abs().amax(1).clamp_min(1e-8) / 127.0
            out[k + ".q8"] = torch.round(v / s[:, None]).clamp(-127, 127).to(torch.int8)
            out[k + ".s8"] = s.half()
        else:
            out[k] = v
    return out


def build(out, weights, vocab, margin=0.0, c=0.0):
    src = {k: base64.b64encode(open(ROOT + v, "rb").read()).decode() for k, v in MODULES.items()}
    sd = quant(torch.load(weights, map_location="cpu"))
    buf = io.BytesIO()
    torch.save(sd, buf)
    w = base64.b64encode(buf.getvalue()).decode()
    voc = base64.b64encode(open(vocab, "rb").read()).decode()
    code = TEMPLATE.replace("__SRC__", repr(src)).replace("__W__", repr(w)).replace("__VOCAB__", repr(voc))
    code = code.replace("__MARGIN__", repr(float(margin))).replace("__C__", repr(float(c)))
    open(out, "w").write(code)
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], *(float(x) for x in sys.argv[4:6]))
