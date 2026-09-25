"""Independent heuristic experts distilled from the leaderboard's top teams (tools/strategy_profile.py,
docs/top_team_profiles.json). They are the ROUTED experts of the H-MoE action layer (model/hmoe.py); the
SHARED expert (always on, as DSV4.1's `shared_experts`) is the strongest complete public agent (cha22).

Each expert is its own decision algorithm for one family of macro decisions and returns a PARTIAL action:
  {"market": [...]}              a complete market order list for this step, and/or
  {"units": {slot_pos: [...]}}   proposals for some unit slots (0 = farmer, i = hand i-1)
Keys that are absent mean "no proposal" (the gate cannot pick the expert for those slots). Experts receive the
shared expert's action as context (like every MoE expert sees the same token): a market expert keeps the
shared plan's orders outside its own concern and replaces the part it owns.

Profiles (from ~1,080 top-team games):
  opening     DSM / Mother-Goose share one exact day-0/1 market sequence (agent/top_openings.json)
  selling     top teams sell continuously from day ~10 around day boundaries (hours 22-2); cha22 sells 57% in
              the last 5 days and dumps fertilizer (48% of its volume at ~1/3 of the price top teams get)
  land        2nd quadrant on day 8-9, 3rd on day 10 (DSM/Mother-Goose/Vadim, 95-100% of games); cha22 11 / 15
  crops       top teams run tomato (~12 tiles) and carrot (5 -> 15 tiles) from mid-game; cha22 ~0
"""
import json
import os

LAND_PRICES = (1000, 2000, 4000)
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
TPD = 24
_OPENINGS = None


def _openings():
    global _OPENINGS
    if _OPENINGS is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "top_openings.json")) as f:
            _OPENINGS = json.load(f)
    return _OPENINGS


def _me(obs):
    p = int(obs["player"])
    return p, obs["farms"][p], obs.get("private") or {}


def _n_animals(farm):
    return sum(1 for row in farm["tiles"] for c in row if isinstance(c, dict) and c.get("animal"))


class Expert:
    name = "expert"

    def __call__(self, obs, shared_action, cfg=None):
        try:
            return self.propose(obs, shared_action or {}) or {}
        except Exception:
            return {}

    def propose(self, obs, shared):
        raise NotImplementedError


class OpeningExpert(Expert):
    """Days 0-1: the exact modal market sequence of a top team."""

    def __init__(self, team="DSM", days=2):
        self.name, self.team, self.days = f"opening_{team}", team, days

    def propose(self, obs, shared):
        t = int(obs["step"])
        seq = _openings()[self.team]
        if t >= min(len(seq), self.days * TPD):
            return {}
        return {"market": [list(o) for o in seq[t]["market"]]}


class SellScheduleExpert(Expert):
    """Sell part of the shed stock at chosen hours from `start_day` on; never dump fertilizer below
    `fert_min_price`; keep a wheat reserve for feeding. Non-SELL orders of the shared plan are kept. On the last
    day the shared plan (terminal liquidation) is left untouched."""

    def __init__(self, name, hours, start_day=10, frac=0.5, fert_min_price=40, wheat_per_animal=3, min_lot=1):
        self.name, self.hours, self.start_day, self.frac = name, set(hours), start_day, frac
        self.fert_min_price, self.wheat_per_animal, self.min_lot = fert_min_price, wheat_per_animal, min_lot

    def propose(self, obs, shared):
        t = int(obs["step"])
        day, hour = t // TPD, t % TPD
        if day < self.start_day or day >= 29:
            return {}
        _, farm, priv = _me(obs)
        base = [o for o in (shared.get("market") or []) if not (isinstance(o, list) and o and o[0] == "SELL")]
        if hour not in self.hours:
            # outside our selling hours: hold, except fertilizer at a good price is kept from the shared plan
            keep = [o for o in (shared.get("market") or []) if isinstance(o, list) and o and o[0] == "SELL"
                    and o[1] == "FERTILIZER" and obs["market"]["prices"].get("FERTILIZER", 0) >= self.fert_min_price]
            return {"market": base + keep}
        shed = priv.get("shed") or {}
        prices = obs["market"]["prices"]
        sells = []
        for item in PRODUCTS:
            stock = int(shed.get(item, 0))
            if item == "WHEAT":
                stock -= self.wheat_per_animal * _n_animals(farm)
            if item == "FERTILIZER" and prices.get(item, 0) < self.fert_min_price:
                continue
            q = int(stock * self.frac)
            if q >= self.min_lot:
                sells.append(["SELL", item, q])
        return {"market": (sells + base)[:10]}


class LandExpert(Expert):
    """Buy the next quadrant on the top teams' schedule if affordable with a cash reserve."""

    def __init__(self, name, days=(6, 9, 10), reserve=300):
        self.name, self.days, self.reserve = name, days, reserve

    def propose(self, obs, shared):
        t = int(obs["step"])
        day = t // TPD
        _, farm, _ = _me(obs)
        k = len(farm["unlocked_quadrants"]) - 1
        market = list(shared.get("market") or [])
        if k >= 3 or day < self.days[k] or any(isinstance(o, list) and o and o[0] == "BUY_LAND" for o in market):
            return {}
        if farm["money"] < LAND_PRICES[k] + self.reserve:
            return {}
        return {"market": [["BUY_LAND"]] + market[:9]}


class CropMixExpert(Expert):
    """Mid/late-game crop diversification into tomato and carrot (hinge-priced, consumed by town shops):
    buys their seeds and, when the shared plan plants a crop, proposes planting the target crop instead."""

    def __init__(self, name, windows=(("TOMATO", 8, 20, 12), ("CARROT", 12, 27, 14)), reserve=500):
        self.name, self.windows, self.reserve = name, windows, reserve

    def _targets(self, day):
        return [(crop, n) for crop, a, b, n in self.windows if a <= day < b]

    def propose(self, obs, shared):
        t = int(obs["step"])
        day = t // TPD
        targets = self._targets(day)
        if not targets:
            return {}
        _, farm, priv = _me(obs)
        seeds = priv.get("seeds") or {}
        planted = {c: 0 for c, _ in targets}
        for row in farm["tiles"]:
            for c in row:
                if isinstance(c, dict) and c.get("crop") in planted:
                    planted[c["crop"]] += 1
        out = {}
        # seeds: keep enough in stock for the remaining target tiles
        market = list(shared.get("market") or [])
        money = farm["money"] - self.reserve
        buys = []
        for crop, n in targets:
            need = max(0, n - planted[crop]) - int(seeds.get(crop, 0))
            q = min(need, int(money // SEED_COST[crop])) if money > 0 else 0
            if q > 0:
                buys.append(["BUY_SEED", crop, q])
                money -= q * SEED_COST[crop]
        if buys:
            out["market"] = (buys + market)[:10]
        # unit slots: re-target PLANT actions of the shared plan
        units = [shared.get("farmer")] + list(shared.get("hands") or [])
        props = {}
        avail = {c: int(seeds.get(c, 0)) for c, _ in targets}
        for pos, u in enumerate(units):
            if isinstance(u, list) and u and u[0] == "PLANT" and (len(u) < 2 or u[1] not in avail):
                for crop, n in targets:
                    if avail[crop] > 0 and planted[crop] < n:
                        props[pos] = ["PLANT", crop]
                        avail[crop] -= 1
                        planted[crop] += 1
                        break
        if props:
            out["units"] = props
        return out


def default_experts():
    """The routed expert set (names are stable indices for the gate)."""
    return [
        OpeningExpert("DSM"),
        OpeningExpert("MMPQ"),
        SellScheduleExpert("sell_DSM", hours=(22, 23, 0, 1, 2), start_day=10, frac=0.5),
        SellScheduleExpert("sell_MMPQ", hours=(23, 0, 1), start_day=10, frac=0.6),
        SellScheduleExpert("sell_Majkel", hours=(18, 21, 22, 0), start_day=9, frac=0.4),
        SellScheduleExpert("sell_late", hours=(22, 23, 0), start_day=24, frac=0.7),
        LandExpert("land_DSM", days=(6, 9, 10)),
        LandExpert("land_MMPQ", days=(6, 8, 13)),
        CropMixExpert("crops_DSM"),
        CropMixExpert("crops_light", windows=(("TOMATO", 10, 20, 6), ("CARROT", 14, 27, 8))),
    ]


def apply(expert_out, shared):
    """Execute an expert fully (gate always picks it where it has a proposal): shared action with the expert's
    market list and unit proposals substituted."""
    act = {"farmer": shared.get("farmer", ["PASS"]), "hands": list(shared.get("hands") or []),
           "market": list(shared.get("market") or [])}
    if "market" in expert_out:
        act["market"] = expert_out["market"]
    for pos, u in (expert_out.get("units") or {}).items():
        if pos == 0:
            act["farmer"] = u
        elif pos - 1 < len(act["hands"]):
            act["hands"][pos - 1] = u
    return act


# ----------------------------------------------------------------------------- demand-aware selling
# Exact market price function of the interpreter (kaggriculture.py MARKET_PARAMS / _r37_market_price):
# price(inv) = base + sign * amp * f(|inv - I0|), amp = target * base / f(T); floor 1. A sold unit is quoted at the
# pre-sale inventory and then adds 1 to it. Premium goods (above_target > 1) crash to the floor on small gluts,
# which is why top teams sell in small lots near base price while cha22 dumps.
import math

MARKET_PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60), "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60), "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40)}
I0 = 10000


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inv):
    base, T, bf, bt, af, at = MARKET_PARAMS[item]
    if inv < I0:
        amp = bt * base / _shape(bf, T, T)
        p = base + amp * _shape(bf, I0 - inv, T)
    else:
        amp = at * base / _shape(af, T, T)
        p = base - amp * _shape(af, inv - I0, T)
    return max(1, int(round(p)))


def sellable(item, inv, want, min_price):
    """How many of `want` units can be sold before the quote drops below min_price (inventory `inv`)."""
    n = 0
    while n < want and market_price(item, inv + n) >= min_price:
        n += 1
    return n


class DemandAwareSellExpert(Expert):
    """mode="clamp": keep the shared plan's SELL timing but cut each lot so the quote stays >= ratio * base
    (opponent_share of the room is left for a rival selling concurrently). mode="spread": additionally sell
    shed stock in small lots whenever the quote is >= ratio * base, from `start_day`. From `free_step` on the
    shared plan (terminal liquidation) is left untouched."""

    def __init__(self, name, ratio=0.9, fert_ratio=0.45, mode="clamp", start_day=10, lot=6,
                 opponent_share=0.3, free_step=24 * 28, wheat_per_animal=3, max_shed=60):
        self.name, self.ratio, self.fert_ratio, self.mode = name, ratio, fert_ratio, mode
        self.start_day, self.lot, self.opp, self.free_step = start_day, lot, opponent_share, free_step
        self.wheat_per_animal = wheat_per_animal
        self.max_shed = max_shed                 # shed capacity is 100 items: never hold back stock when it is filling

    def _min_price(self, item):
        base = MARKET_PARAMS[item][0]
        return (self.fert_ratio if item == "FERTILIZER" else self.ratio) * base

    def propose(self, obs, shared):
        t = int(obs["step"])
        if t >= self.free_step:
            return {}
        _, farm, priv = _me(obs)
        shed_now = sum(int(v) for v in (priv.get("shed") or {}).values())
        carried = sum(int(v) for inv_u in (priv.get("inventories") or []) for v in (inv_u or {}).values())
        if shed_now + carried >= self.max_shed:           # storage pressure: the shared plan's dumping is needed
            return {}
        inv = dict(obs["market"].get("inventory") or {})
        market = [list(o) for o in (shared.get("market") or []) if isinstance(o, list)]
        out, changed = [], False
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in MARKET_PARAMS:
                want = int(o[2]) if isinstance(o[2], (int, float)) else 0
                room = sellable(o[1], inv.get(o[1], I0), want, self._min_price(o[1]))
                q = int(math.floor(room * (1 - self.opp))) if room < want else want
                if q != want:
                    changed = True
                if q > 0:
                    out.append(["SELL", o[1], q])
                    inv[o[1]] = inv.get(o[1], I0) + q
            else:
                out.append(o)
        if self.mode == "spread" and t // TPD >= self.start_day:
            _, farm, priv = _me(obs)
            shed = priv.get("shed") or {}
            planned = {o[1]: o[2] for o in out if o and o[0] == "SELL"}
            for item in PRODUCTS:
                stock = int(shed.get(item, 0)) - int(planned.get(item, 0))
                if item == "WHEAT":
                    stock -= self.wheat_per_animal * _n_animals(farm)
                if stock <= 0:
                    continue
                q = min(self.lot, stock, sellable(item, inv.get(item, I0), self.lot, self._min_price(item)))
                if q > 0:
                    out.append(["SELL", item, q])
                    inv[item] = inv.get(item, I0) + q
                    changed = True
        return {"market": out[:10]} if changed else {}


def demand_experts():
    return [
        DemandAwareSellExpert("dsell_clamp90_shed40", ratio=0.9, mode="clamp", max_shed=40),
        DemandAwareSellExpert("dsell_fert_shed40", ratio=0.0, fert_ratio=0.45, mode="clamp", max_shed=40),
        DemandAwareSellExpert("dsell_clamp80_shed60", ratio=0.8, mode="clamp", max_shed=60),
        DemandAwareSellExpert("dsell_clamp90", ratio=0.9, mode="clamp"),
        DemandAwareSellExpert("dsell_clamp100", ratio=1.0, mode="clamp"),
        DemandAwareSellExpert("dsell_fert_only", ratio=0.0, fert_ratio=0.45, mode="clamp"),
        DemandAwareSellExpert("dsell_spread95", ratio=0.95, mode="spread", start_day=10, lot=4),
        DemandAwareSellExpert("dsell_spread105", ratio=1.05, mode="spread", start_day=8, lot=3),
    ]
