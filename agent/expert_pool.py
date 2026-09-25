"""Expert pool of the H-MoE policy (DeepSeek-V4.1 MoE layout: one shared expert + routed experts).

  shared   cha22 (strongest complete public agent), always on
  routed   independent complete agents from other sources (league/pub/*, agent/planner.py), top-team
           strategy experts (agent/top_experts.py), and prefixes of cha22's patch-layer stack (skip later layers)

`propose(obs, cfg)` runs every expert in shadow mode (their internal state follows the real game) and returns
the shared action plus one partial action per routed expert. `slot_table(...)` turns them into per-slot
candidate indices: -1 where an expert has no proposal (it is then excluded from gating for that slot).
"""
import time

from agent.action_space import SLOT_MARKET, STOP, UNIT_NONE, order_to_cand, unit_to_cand
from agent.loader import load_agent
from agent.planner import TopPlanner
from agent.skills import LayeredExpert
from agent.top_experts import (CropMixExpert, DemandAwareSellExpert, LandExpert, OpeningExpert,
                               SellScheduleExpert)

PUB = ["prvsiyan_frontier", "hakfield", "tetsutani_ms", "pilkwang_sep", "dmitrii_2c1s"]
CHA22_LAYERS = ["_IG_PARENT", "_MG_PARENT", "_E402_PARENT", "_E410_PARENT"]   # skip the last patches


def top_experts():
    return [OpeningExpert("DSM"), OpeningExpert("MMPQ"),
            LandExpert("land_DSM", days=(6, 9, 10)), LandExpert("land_MMPQ", days=(6, 8, 13)),
            CropMixExpert("crops_light", windows=(("TOMATO", 10, 20, 6), ("CARROT", 14, 27, 8))),
            CropMixExpert("crops_DSM"),
            SellScheduleExpert("sell_late", hours=(22, 23, 0), start_day=24, frac=0.7),
            DemandAwareSellExpert("dsell_fert_shed40", ratio=0.0, fert_ratio=0.45, mode="clamp", max_shed=40),
            DemandAwareSellExpert("dsell_clamp90_shed40", ratio=0.9, mode="clamp", max_shed=40),
            DemandAwareSellExpert("dsell_clamp80_shed60", ratio=0.8, mode="clamp", max_shed=60)]


class ExpertPool:
    def __init__(self, root="league", pub=PUB, layers=CHA22_LAYERS, planner=True, tops=True):
        self.shared = LayeredExpert(f"{root}/cha22.py", name="cha22", record_layers=bool(layers))
        self.layer_names = [l for l in layers if l in self.shared.layers]
        self.whole = [(n, load_agent(f"{root}/pub/{n}/main.py")) for n in pub]
        if planner:
            self.whole.append(("planner", TopPlanner()))
        self.tops = top_experts() if tops else []
        self.names = ([f"cha22:{l}" for l in self.layer_names] + [n for n, _ in self.whole] +
                      [e.name for e in self.tops])
        self.times = []

    def __len__(self):
        return len(self.names)

    def propose(self, obs, cfg=None):
        """-> (shared full action, [partial action per routed expert])."""
        t0 = time.time()
        shared, layers = self.shared(obs, cfg)
        got = dict(layers)
        out = [got.get(l, shared) for l in self.layer_names]
        for n, f in self.whole:
            try:
                a = f(obs, cfg)
            except Exception:
                a = None
            out.append(a if isinstance(a, dict) else {})
        for e in self.tops:
            out.append(e(obs, shared, cfg))
        self.times.append(time.time() - t0)
        return shared, out


def _units_of(a, n_hands):
    """Unit proposals of a (partial) action: list of length 1+n_hands with raw items or None."""
    if not isinstance(a, dict):
        return [None] * (1 + n_hands)
    if "units" in a:                                          # partial expert: {"units": {pos: item}}
        return [a["units"].get(i) for i in range(1 + n_hands)]
    if "farmer" not in a and "hands" not in a:
        return [None] * (1 + n_hands)
    hands = a.get("hands") if isinstance(a.get("hands"), list) else []
    return [a.get("farmer")] + [hands[i] if i < len(hands) else ["PASS"] for i in range(n_hands)]


def slot_table(shared, partials, n_hands, max_orders=10):
    """-> dict with per-slot candidates and raw items:
       unit_c [1+n_hands][E], unit_raw, unit_shared (cand, raw);
       market_c [max_orders+1][E] (STOP past a list's end, -1 without a market proposal), market_raw, market_shared."""
    E = len(partials)
    su = _units_of(shared, n_hands)
    unit_c, unit_raw = [], []
    pu = [_units_of(p, n_hands) for p in partials]
    for pos in range(1 + n_hands):
        unit_c.append([unit_to_cand(pu[k][pos]) if pu[k][pos] is not None else -1 for k in range(E)])
        unit_raw.append([pu[k][pos] for k in range(E)])
    sm = (shared.get("market") or [])[:max_orders]
    pm = [(p.get("market")[:max_orders] if isinstance(p, dict) and isinstance(p.get("market"), list) else None)
          for p in partials]
    market_c, market_raw = [], []
    for j in range(max_orders + 1):
        market_c.append([(-1 if m is None else (order_to_cand(m[j]) if j < len(m) else STOP)) for m in pm])
        market_raw.append([(m[j] if m is not None and j < len(m) else None) for m in pm])
    return {"unit_c": unit_c, "unit_raw": unit_raw,
            "unit_shared": [(unit_to_cand(su[pos]) if su[pos] is not None else UNIT_NONE, su[pos])
                            for pos in range(1 + n_hands)],
            "market_c": market_c, "market_raw": market_raw,
            "market_shared": [((order_to_cand(sm[j]) if j < len(sm) else STOP), (sm[j] if j < len(sm) else None))
                              for j in range(max_orders + 1)]}
