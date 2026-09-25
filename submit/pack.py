"""Pack a self-contained single-file Kaggle submission (main.py).

main.py = base executor source (exec'd into its own module) + features + numpy TTT policy
        + controller + base64 weights. No torch / scipy needed at inference.

python -m submit.pack <weights.npz|none> <out_main.py> [base_agent.py]
"""
import base64
import io
import sys

import numpy as np

ROOT = "/home/user/kgc/"


def strip_imports(src, drop):
    out = []
    for line in src.splitlines():
        if any(line.startswith(d) for d in drop):
            continue
        out.append(line)
    return "\n".join(out)


def build(weights, out, base=ROOT + "league/metav4.py"):
    base_src = open(base).read()
    feats = open(ROOT + "agent/features.py").read()
    pol = strip_imports(open(ROOT + "model/policy_np.py").read(), ["from agent.features import"])
    ctrl = strip_imports(open(ROOT + "agent/controller.py").read(), ["from agent.features import"])
    if weights and weights != "none":
        buf = io.BytesIO()
        np.savez_compressed(buf, **dict(np.load(weights)))
        wb64 = base64.b64encode(buf.getvalue()).decode()
    else:
        wb64 = ""
    code = f'''# Kaggriculture agent: base executor + TTT/Transformer RL market controller.
import base64 as _b64, io as _io, sys as _sys, types as _types
import numpy as np

_BASE_SRC = {base_src!r}
_base_mod = _types.ModuleType("base_executor")
_base_mod.__dict__["__name__"] = "base_executor"
exec(compile(_BASE_SRC, "base_executor.py", "exec"), _base_mod.__dict__)
_base_agent = [v for k, v in _base_mod.__dict__.items() if callable(v) and not k.startswith("__")][-1]   # Kaggle entry rule

# ---------------- features ----------------
{feats}

# ---------------- numpy policy ----------------
{pol}

# ---------------- controller ----------------
{ctrl}

_WB64 = {wb64!r}
_CTRL = None


def _make():
    global _CTRL
    if _WB64:
        w = dict(np.load(_io.BytesIO(_b64.b64decode(_WB64))))
        _pol = NumpyPolicy(w)
        _CTRL = MarketController(_base_agent, _pol.act_greedy)
        _CTRL._pol = _pol
    else:
        _CTRL = MarketController(_base_agent, None)
    return _CTRL


def agent(observation, configuration=None):
    global _CTRL
    if _CTRL is None or int(observation["step"]) == 0:
        _make()
    return _CTRL(observation, configuration)   # controller falls back to the base action on errors
'''
    with open(out, "w") as f:
        f.write(code)
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
