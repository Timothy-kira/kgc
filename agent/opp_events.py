"""Opponent event streams for the Engram opponent memory (docs/PLAN_v4.1.md).

S stream: one raw code per game step = (hour block, signed flow bucket of the opponent's inventory-moving market
          flow for each of the 9 products, visible farm change class: crops planted, animals added, hires, land).
D stream: one raw code per day = (signed daily flow bucket per product, animals added per type, hires, land),
          preceded by a fingerprint code at step 2 (opponent money + market wheat, cha22's CTRTABLE key).
Raw codes are int64 mixed-radix values; model/engram training maps them to a compressed vocabulary.

`farm_summary` / `farm_change` read only the opponent's public farm. `infer_flows` recovers the opponent's
flows from what a live agent observes (public inventory + shops, own orders, own unit phase replayed with the
official rules; 99.87% exact, tools/opp_flow_check.py). Offline extraction may pass the engine's exact flows.
"""
import collections
import copy

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHOPS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
         "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
         "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
         "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
TOWN_CENTER = PRODUCTS[:-1]
STEP_EDGES = (1, 3, 6, 11)          # |q| buckets: 0 | 1-2 | 3-5 | 6-10 | 11+
DAY_EDGES = (1, 6, 16, 41)          # |q| buckets per day: 0 | 1-5 | 6-15 | 16-40 | 41+
FP_FLAG = 1 << 62


def signed_bucket(q, edges):
    a = abs(int(q))
    b = sum(a >= e for e in edges)
    return 4 + (b if q > 0 else -b)            # 0..8, 4 = zero


def drain(step, shops):
    d = collections.Counter()
    if step % 4 == 0:
        for s in shops:
            prods = SHOPS.get(s, ())
            for it in prods:
                d[it] += 2 if len(prods) == 1 else 1
    if step % 24 == 0:
        for it in TOWN_CENTER:
            d[it] += 1
    return d


def farm_summary(farm):
    """Counts on a public farm: plants per crop, animals per type, hands, unlocked quadrants."""
    plants, animals = collections.Counter(), collections.Counter()
    for row in farm["tiles"]:
        for c in row:
            if isinstance(c, dict):
                if c.get("kind") == "PLANT":
                    plants[c.get("crop")] += 1
                if c.get("animal"):
                    animals[c.get("animal")] += 1
    return plants, animals, len(farm.get("hands") or []), len(farm.get("unlocked_quadrants") or [])


def farm_change(prev, cur):
    """(crop-planted bitmask, animal-added bitmask, hires bucket 0..3, land bit) between two summaries."""
    p0, a0, h0, q0 = prev
    p1, a1, h1, q1 = cur
    crops = sum(1 << i for i, c in enumerate(CROPS) if p1[c] > p0[c])
    anim = sum(1 << i for i, a in enumerate(ANIMALS) if a1[a] > a0[a])
    dh = max(0, h1 - h0)
    return crops, anim, (0 if dh == 0 else 1 if dh <= 3 else 2 if dh <= 7 else 3), int(q1 > q0)


def s_code(step, flows, change):
    code = (step % 24) // 6
    for p in PRODUCTS:
        code = code * 9 + signed_bucket(flows.get(p, 0), STEP_EDGES)
    crops, anim, hires, land = change
    return ((code * 32 + crops) * 8 + anim) * 8 + hires * 2 + land


def d_code(day_flows, animals_added, hires, land):
    code = 0
    for p in PRODUCTS:
        code = code * 9 + signed_bucket(day_flows.get(p, 0), DAY_EDGES)
    for a in ANIMALS:
        code = code * 4 + min(3, animals_added.get(a, 0))
    return (code * 4 + (0 if hires == 0 else 1 if hires <= 20 else 2 if hires <= 60 else 3)) * 2 + int(land)


def fingerprint_code(opp_money, market_wheat):
    return FP_FLAG | ((int(round(opp_money)) & 0xFFFFF) << 20) | (int(market_wheat) & 0xFFFFF)


class OppEventStream:
    """Accumulates one opponent's S codes (per step) and D codes (per day, fingerprint first)."""

    def __init__(self):
        self.s, self.d = [], []
        self.day_flow, self.day_anim = collections.Counter(), collections.Counter()
        self.day_hires, self.day_land = 0, 0
        self.prev_summary = None

    def push(self, step, flows, opp_farm, opp_money=None, market_wheat=None):
        """Record the opponent's events of transition `step` (flows: {product: signed inventory-moving qty};
        opp_farm: the opponent's public farm after the transition)."""
        summ = farm_summary(opp_farm)
        change = farm_change(self.prev_summary, summ) if self.prev_summary else (0, 0, 0, 0)
        if self.prev_summary:
            for a in ANIMALS:
                self.day_anim[a] += max(0, summ[1][a] - self.prev_summary[1][a])
            self.day_hires += max(0, summ[2] - self.prev_summary[2])
            self.day_land |= int(summ[3] > self.prev_summary[3])
        self.prev_summary = summ
        self.s.append(s_code(step, flows, change))
        for p, q in flows.items():
            self.day_flow[p] += q
        if step == 2 and opp_money is not None:
            self.d.append(fingerprint_code(opp_money, market_wheat))
        if step % 24 == 23:
            self.d.append(d_code(self.day_flow, self.day_anim, self.day_hires, self.day_land))
            self.day_flow, self.day_anim = collections.Counter(), collections.Counter()
            self.day_hires, self.day_land = 0, 0


def infer_flows(K, obs_before, my_action, inv_after):
    """Opponent's inventory-moving flow of one transition from a live agent's view (see module doc).
    K: the kaggriculture module (official rules), obs_before: our observation before the transition,
    my_action: our action for it, inv_after: public market inventory after it."""
    me = int(obs_before["player"])
    inv_before = obs_before["market"]["inventory"]
    step = int(obs_before["step"])
    farm_c = copy.deepcopy(obs_before["farms"][me])
    priv_c = copy.deepcopy(obs_before["private"])
    units = [my_action.get("farmer", ["PASS"])] + list(my_action.get("hands") or [])
    demand = collections.Counter(u[1] for u in units if isinstance(u, list) and len(u) >= 2 and u[0] == "PLANT")
    seeds = priv_c.get("seeds", {})
    blocked = {c for c, n in demand.items() if n > seeds.get(c, 0)}
    for i, u in enumerate(units):
        if isinstance(u, list) and len(u) >= 2 and u[0] == "PLANT" and u[1] in blocked:
            u = ["PASS"]
        try:
            K._apply_unit_action(farm_c, priv_c, i, u, 10, obs_before["day"], 24, 100)
        except Exception:
            pass
    ours = collections.Counter()
    left, inv_sim = dict(priv_c["shed"]), dict(inv_before)
    money, room = farm_c["money"], 100 - sum(priv_c["shed"].values())
    for m in my_action.get("market") or []:
        if not (isinstance(m, list) and len(m) >= 3 and m[1] in PRODUCTS):
            continue
        it, q = m[1], int(m[2])
        if m[0] == "SELL":
            for _ in range(min(q, left.get(it, 0))):
                left[it] -= 1
                pr = K.market_price(it, inv_sim[it])
                money += pr
                room += 1
                if pr > 1:
                    inv_sim[it] += 1
                    ours[it] += 1
        elif m[0] == "BUY_PRODUCT":
            for _ in range(q):
                pr = K.market_price(it, inv_sim[it] - 1)
                if money < pr or room <= 0:
                    break
                money -= pr
                room -= 1
                inv_sim[it] -= 1
                ours[it] -= 1
    dr = drain(step, obs_before["town"]["unlocked_shops"])
    return {p: (inv_after[p] - inv_before[p]) + dr[p] - ours[p] for p in PRODUCTS}


# ----------------------------------------------------------------------------- CLM-style opponent targets
HORIZONS = (1, 4, 24, 72, 0)                     # steps ahead; 0 = until the end of the game
H_EDGES = {1: STEP_EDGES, 4: STEP_EDGES, 24: DAY_EDGES, 72: (1, 11, 41, 121), 0: (1, 11, 41, 121)}
N_BUCKET = 9
DESC_DIM = len(PRODUCTS) + 3 + 1 + len(HORIZONS)


def opp_targets(flow):
    """flow int [T, 9] (opponent's flow of transition t) -> bucket ids int64 [T, len(HORIZONS), 9]: for the
    decision at step t (which has seen transitions < t) the summed flow over transitions t .. t+h-1."""
    import numpy as np
    flow = np.asarray(flow, np.int64)
    T = len(flow)
    cs = np.concatenate([np.zeros((1, flow.shape[1]), np.int64), np.cumsum(flow, 0)])
    out = np.empty((T, len(HORIZONS), flow.shape[1]), np.int64)
    idx = np.arange(T)
    for j, h in enumerate(HORIZONS):
        end = np.full(T, T) if h == 0 else np.minimum(idx + h, T)
        q = cs[end] - cs[idx]
        edges = np.array(H_EDGES[h])
        b = (np.abs(q)[..., None] >= edges).sum(-1)
        out[:, j] = 4 + np.sign(q) * b
    return out


def opp_desc():
    """Candidate descriptors float32 [len(HORIZONS), 9 products, 9 buckets, DESC_DIM] for the CLM opponent head:
    product one-hot, sign one-hot, log magnitude of the bucket's lower edge, horizon one-hot."""
    import math
    import numpy as np
    d = np.zeros((len(HORIZONS), len(PRODUCTS), N_BUCKET, DESC_DIM), np.float32)
    for j, h in enumerate(HORIZONS):
        edges = (0,) + tuple(H_EDGES[h])
        for p in range(len(PRODUCTS)):
            for b in range(N_BUCKET):
                k = b - 4
                d[j, p, b, p] = 1
                d[j, p, b, len(PRODUCTS) + (0 if k < 0 else 1 if k == 0 else 2)] = 1
                d[j, p, b, len(PRODUCTS) + 3] = math.log1p(edges[abs(k)]) / 5.0
                d[j, p, b, len(PRODUCTS) + 4 + j] = 1
    return d
