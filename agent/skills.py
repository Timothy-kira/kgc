"""Heuristic agents decomposed into fine-grained skill experts (the routed experts of the H-MoE action layer).

Public Kaggriculture agents are stacks of patch layers: each layer does
    _X_PARENT = agent
    def agent(obs, cfg): a = _X_PARENT(obs, cfg); <modify a>; return a
and looks its parent up by global name at call time. Wrapping every `_X_PARENT` global with a recorder makes a
single call of the full agent yield the action after *every prefix of the stack* - i.e. one proposal per layer
("skill") for free. Each layer is one expert; the router (model/hmoe.py) decides, per decision slot, which
layer's proposal to follow, so every heuristic layer becomes controllable by the policy / RL.

Experts are run in shadow mode: every expert is called every step with the true observation (keeps their
internal trackers consistent even when another expert's action was executed).
"""
import copy
import functools
import importlib.util
import inspect
import re
import sys
import time
import uuid

from agent.action_space import SLOT_MARKET, STOP, action_to_decisions

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
_PARENT_RE = re.compile(r"^(_[A-Za-z0-9_]+)\s*=\s*agent\s*$", re.M)


def _load(path):
    name = "skill_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class LayeredExpert:
    """One heuristic agent file; call -> (final action, [(layer_name, action_before_layer_i+1), ...])."""

    def __init__(self, path, name=None, record_layers=True):
        self.path, self.name = path, name or path.rsplit("/", 1)[-1].replace(".py", "")
        self.m = _load(path)
        fn = self.m.agent
        try:
            self.n_args = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            self.n_args = 2
        self.layers = []
        self._rec = {}
        if record_layers:
            src = open(path).read()
            for g in dict.fromkeys(_PARENT_RE.findall(src)):          # source order = stack order
                f = getattr(self.m, g, None)
                if callable(f):
                    setattr(self.m, g, self._recorder(g, f))
                    self.layers.append(g)

    def _recorder(self, name, f):
        rec = self._rec

        def wrapped(*a, **k):
            out = f(*a, **k)
            rec[name] = copy.deepcopy(out) if isinstance(out, dict) else out
            return out
        # layers read attributes of their parent (e.g. `.chassis` set by make_agent): keep them visible
        functools.update_wrapper(wrapped, f)
        return wrapped

    def __call__(self, obs, cfg=None):
        self._rec.clear()
        try:
            act = self.m.agent(obs, cfg) if self.n_args >= 2 else self.m.agent(obs)
        except Exception:
            act = PASS
        act = act if isinstance(act, dict) else PASS
        layers = [(n, self._rec[n]) for n in self.layers if isinstance(self._rec.get(n), dict)]
        return act, layers


class SkillPool:
    """All experts (whole agents + their layer prefixes). `propose(obs, cfg)` -> list of (skill_name, action).
    Skills whose action equals the previous layer's are kept (the router needs a stable expert index), but
    `distinct()` tells which actually differ."""

    def __init__(self, paths, record_layers=True):
        self.experts = [LayeredExpert(p, record_layers=record_layers) for p in paths]
        self.names = []
        for e in self.experts:
            self.names += [f"{e.name}:{n}" for n in e.layers] + [f"{e.name}:final"]
        self.times = []

    def propose(self, obs, cfg=None):
        t0 = time.time()
        out = []
        for e in self.experts:
            final, layers = e(obs, cfg)
            got = dict(layers)
            for n in e.layers:                      # a layer not reached this step (early-return path): final
                out.append((f"{e.name}:{n}", got.get(n, final)))
            out.append((f"{e.name}:final", final))
        self.times.append(time.time() - t0)
        return out


def slot_proposals(actions, n_hands):
    """actions: list of full actions (one per skill) -> per skill the list of (slot, candidate) decisions, and
    a slot-aligned table: table[j] = candidate of each skill at decision position j of that skill's own plan.
    Unit slots (farmer, hand i) align by position; market slot j aligns with the skill's j-th order (STOP once
    its list is exhausted)."""
    decs = [action_to_decisions(a, n_hands) for a in actions]
    n_unit = 1 + n_hands
    units = [[c for s, c in d[:n_unit]] for d in decs]
    markets = [[c for s, c in d[n_unit:] if s == SLOT_MARKET] for d in decs]
    return units, markets


def market_proposal(markets, j):
    """Candidate each skill proposes for market slot j (STOP past the end of its list)."""
    return [m[j] if j < len(m) else STOP for m in markets]
