"""Pack the v4 agent (cha22 skeleton + Engram insertion points) into a single Kaggle main.py.

python -m submit.pack_v4 <out_main.py> [route_weights.pt|none] [chassis settings json, e.g. '{"clamp_sells": true}']
Embedded: league/cha22.py (exec'd as its own module), model/engram.py, model/route_engram.py,
agent/opp_events.py. Runtime per step: the opponent's events are inferred from the observation
(agent/opp_events.infer_flows) and pushed into the Engram key stream; at step 144 cha22's route table choice is
replaced by the Engram route memory when it predicts a gain above the margin (untrained: never, so the agent is
bit-identical to cha22). Any exception falls back to cha22's own action.
"""
import base64
import sys

ROOT = "/home/user/kgc/"
MODULES = {"engram": "model/engram.py", "route_engram": "model/route_engram.py", "opp_events": "agent/opp_events.py",
           "cha22_base": "league/cha22.py"}

TEMPLATE = r'''# Kaggriculture v4: cha22 skeleton + Engram insertion points (H5 route memory; opponent event stream).
import base64 as _b64, sys as _sys, types as _types, io as _io
_SRC = __SRC__
_W = __W__
_SETTINGS = __SETTINGS__

def _load(name, deps=()):
    m = _types.ModuleType(name)
    m.__dict__["__name__"] = name
    _sys.modules[name] = m
    exec(compile(_b64.b64decode(_SRC[name]).decode(), name + ".py", "exec"), m.__dict__)
    return m

# the embedded modules import each other as package paths
for _pkg in ("model", "agent"):
    if _pkg not in _sys.modules:
        _sys.modules[_pkg] = _types.ModuleType(_pkg)
_eng = _load("model.engram"); _sys.modules["model"].engram = _eng
_opp = _load("agent.opp_events"); _sys.modules["agent"].opp_events = _opp
_re = _load("model.route_engram")
_base = _load("cha22_base")

def _entry(m):
    """Kaggle's rule (agent/loader.entry_name): the last callable binding that does not start with '__'."""
    fn = None
    for k, v in list(m.__dict__.items()):
        if callable(v) and not k.startswith("__"):
            fn = v
    return fn

_BASE_FN = _entry(_base)
try:
    import importlib as _il
    _K = _il.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
except Exception:
    _K = None

class _V4:
    def __init__(self):
        self.stream = _opp.OppEventStream()
        self.prev = None
        self.prev_action = None
        self.model = None
        self.route_log = None
        try:
            import torch
            torch.set_num_threads(1)
            ch = _base._IMPL.chassis
            ch.cfg.update(_SETTINGS)          # knob insertion point (tools/settings_sweep.py)
            routes = sorted(ch.routes)
            if _W:
                blob = torch.load(_io.BytesIO(_b64.b64decode(_W)), map_location="cpu", weights_only=False)
                vocab = _re.RouteVocab(codes=[])
                vocab.keys = blob["vocab_keys"]
                vocab.size = 2 + vocab.n_shop + len(vocab.keys)
                self.model = _re.RouteEngram(vocab.size, blob["routes"])
                self.model.load_state_dict(blob["state"])
            else:
                vocab = _re.RouteVocab(codes=[])
                torch.manual_seed(0)
                self.model = _re.RouteEngram(vocab.size, routes)
            self.model.eval()
            self.vocab = vocab
            base_router = ch.router
            agent = self
            def router(obs, step, st):
                r = base_router(obs, step, st)
                if step >= 144 and step < 648 and "v4_route" not in st:
                    st["v4_route"] = r
                    try:
                        shops2 = list((obs["town"]["unlocked_shops"] or [])[:2])
                        seq = agent.vocab.sequence(agent.stream.d, shops2)
                        choice, gain = agent.model.choose(seq, r)
                        st["v4_route"] = choice
                        agent.route_log = (r, choice, gain)
                    except Exception:
                        st["v4_route"] = r
                if 144 <= step < 648 and "v4_route" in st:
                    return st["v4_route"]
                return r
            ch.router = router
        except Exception:
            self.model = None

    def __call__(self, obs, config=None):
        try:
            if self.prev is not None and _K is not None:
                me = int(obs["player"])
                flows = _opp.infer_flows(_K, self.prev, self.prev_action, obs["market"]["inventory"])
                o = obs["farms"][1 - me]
                self.stream.push(int(self.prev["step"]), flows, o, opp_money=o["money"],
                                 market_wheat=obs["market"]["inventory"]["WHEAT"])
        except Exception:
            pass
        action = _BASE_FN(obs, config) if _BASE_FN.__code__.co_argcount > 1 else _BASE_FN(obs)
        try:
            import copy as _c
            self.prev, self.prev_action = _c.deepcopy(obs), _c.deepcopy(action)
        except Exception:
            self.prev = None
        return action

_AGENT = None

def agent(obs, config=None):
    global _AGENT
    if _AGENT is None:
        _AGENT = _V4()
    return _AGENT(obs, config)
'''


def build(out, weights=None, settings=None):
    src = {k: base64.b64encode(open(ROOT + v, "rb").read()).decode() for k, v in MODULES.items()}
    src["model.engram"] = src.pop("engram")
    src["agent.opp_events"] = src.pop("opp_events")
    src["model.route_engram"] = src.pop("route_engram")
    w = base64.b64encode(open(weights, "rb").read()).decode() if weights and weights != "none" else ""
    code = TEMPLATE.replace("__SRC__", repr(src)).replace("__W__", repr(w)).replace("__SETTINGS__", repr(settings or {}))
    open(out, "w").write(code)
    return out


if __name__ == "__main__":
    import json as _json
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
          _json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
