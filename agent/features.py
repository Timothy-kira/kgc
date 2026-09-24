"""Observation -> tensors for the market-decision policy (TTT + Transformer).

Pure numpy / stdlib so it runs inside the Kaggle submission. One `Tracker`
instance per (game, player); call `update(obs)` once per step *before*
choosing an action, then `features(base_market)`.

Tokens per step
    product tokens  [9, PF]  one per sellable product
    global token    [GF]
Self-supervised TTT targets (available at the end of every day)
    per-product net market flow over the day (= all player sales - buys),
    per-product log price change over the day.
"""
import math

import numpy as np

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
PIDX = {p: i for i, p in enumerate(PRODUCTS)}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160,
        "WOOL": 200, "FERTILIZER": 100}
TPARAM = {"WHEAT": 400, "CARROT": 450, "TOMATO": 200, "STRAWBERRY": 100, "MELON": 300, "EGG": 332, "MILK": 122,
          "WOOL": 105, "FERTILIZER": 200}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
I0 = 10000
EPISODE = 720
TPD = 24

# Sell action bins (fraction of current stock = shed + units' carried inventory).
# The last bin is "sell everything available" (order size 999; SELL stops when the shed is empty).
# Bin 5 = "BASE": keep the base executor's own SELL orders for that product verbatim
# (the base agent's quantities are finely tuned; the default policy is exactly the base agent).
SELL_BINS = [0.0, "ONE", 0.25, 0.5, 1.0, "BASE"]
N_BINS = len(SELL_BINS)
BASE_BIN = 5

PF = 32   # product feature width
GF = 16   # global feature width
N_TGT = 2 * len(PRODUCTS)  # TTT self-supervised targets per day

try:  # official price function when available (local training); fallback copy for submission
    from kaggle_environments.envs.kaggriculture.kaggriculture import market_price as _market_price
except Exception:  # pragma: no cover
    _PARAMS = {
        "WHEAT": (25, 400, "sqrt", .8, "log", .2), "CARROT": (35, 450, "hinge", 1., "sqrt", .7),
        "TOMATO": (60, 200, "hinge", .4, "sqrt", .6), "STRAWBERRY": (120, 100, "sqrt", .7, "linear", 1.6),
        "MELON": (250, 300, "log", .2, "sq", 3.6), "EGG": (50, 332, "hinge", .4, "log", .2),
        "MILK": (160, 122, "sqrt", .6, "linear", 1.6), "WOOL": (200, 105, "log", .2, "sq", 3.2),
        "FERTILIZER": (100, 200, "linear", .4, "linear", .4)}

    def _shape(f, x, T):
        x = max(0.0, x)
        if f == "linear": return x
        if f == "sq": return x * x
        if f == "sqrt": return math.sqrt(x)
        if f == "log": return math.log(1.0 + x)
        if f == "hinge":
            u = x / T
            return u + 8.0 * max(0.0, u - 1.0) ** 2
        return x

    def _market_price(item, inv, params=None):
        b, T, bf, bt, af, at = _PARAMS[item]
        if inv < I0:
            p = b + bt * b / _shape(bf, T, T) * _shape(bf, I0 - inv, T)
        else:
            p = b - at * b / _shape(af, T, T) * _shape(af, inv - I0, T)
        return max(1, int(round(p)))


def market_price(item, inv):
    return _market_price(item, inv)


def town_consumption(shops, step):
    """Units of each product the town removes at `step` (turn processing step 4)."""
    out = np.zeros(len(PRODUCTS))
    if step % 4 == 0:
        for s in shops:
            prods = SHOPS.get(s, [])
            mult = 2 if len(prods) == 1 else 1
            for p in prods:
                out[PIDX[p]] += mult
    if step % 24 == 0:
        out[:-1] += 1
    return out


def shop_rate(shops):
    """Expected per-day town demand per product."""
    r = np.zeros(len(PRODUCTS))
    for s in shops:
        prods = SHOPS.get(s, [])
        mult = 2 if len(prods) == 1 else 1
        for p in prods:
            r[PIDX[p]] += 6 * mult
    r[:-1] += 1
    return r


def _farm_stats(farm, day):
    """Per-product harvestable units on tiles, and growing units expected soon."""
    ready = np.zeros(len(PRODUCTS))
    growing = np.zeros(len(PRODUCTS))
    animals = np.zeros(len(PRODUCTS))
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                k = PIDX[t["crop"]]
                ready[k] += t.get("yield_units", 0)
                growing[k] += 1
            elif t.get("animal"):
                k = PIDX[ANIMAL_PRODUCT[t["animal"]]]
                ready[k] += t.get("yield_units", 0)
                animals[k] += 1
    return ready, growing, animals


class Tracker:
    def __init__(self):
        self.prev_inv = None
        self.prev_prices = None
        self.hist_flow = []      # per-step net flow vectors (player sales - buys)
        self.day_flow = np.zeros(len(PRODUCTS))
        self.day_start_price = None
        self.day_targets = []    # list of (day, target vector)
        self.last_step = -1
        self.my_last_orders = np.zeros(len(PRODUCTS))

    def update(self, obs):
        step = int(obs["step"])
        inv = obs["market"]["inventory"]
        prices = obs["market"]["prices"]
        shops = obs["town"]["unlocked_shops"]
        invv = np.array([inv[p] for p in PRODUCTS], dtype=np.float64)
        pv = np.array([prices[p] for p in PRODUCTS], dtype=np.float64)
        if self.prev_inv is not None and step == self.last_step + 1:
            # inventory(t) = inventory(t-1) + sales - buys - town(t-1)
            flow = invv - self.prev_inv + town_consumption(shops, step - 1)
            self.hist_flow.append(flow)
            self.day_flow += flow
        else:
            self.hist_flow.append(np.zeros(len(PRODUCTS)))
        if step % TPD == 0:
            if self.day_start_price is not None:
                tgt = np.concatenate([np.sign(self.day_flow) * np.log1p(np.abs(self.day_flow)),
                                      np.log(pv) - np.log(self.day_start_price)])
                self.day_targets.append((step // TPD - 1, tgt))
            self.day_flow = np.zeros(len(PRODUCTS))
            self.day_start_price = pv
        if self.day_start_price is None:
            self.day_start_price = pv
        self.prev_inv, self.prev_prices, self.last_step = invv, pv, step
        self.obs = obs

    def stock(self):
        """Units per product available to sell this turn (shed + units' carried inventory)."""
        priv = self.obs["private"]
        s = np.array([priv["shed"].get(p, 0) for p in PRODUCTS], dtype=np.float64)
        for inv in priv.get("inventories", []):
            for p, n in inv.items():
                if p in PIDX:
                    s[PIDX[p]] += n
        return s

    def features(self, base_market=None):
        obs = self.obs
        me = int(obs["player"])
        step = int(obs["step"])
        day = step // TPD
        farms = obs["farms"]
        inv = obs["market"]["inventory"]
        prices = obs["market"]["prices"]
        shops = obs["town"]["unlocked_shops"]
        shed = obs["private"]["shed"]
        stock = self.stock()
        base_q = np.zeros(len(PRODUCTS))
        for o in (base_market or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in PIDX:
                base_q[PIDX[o[1]]] += float(o[2])
        rd_m, gr_m, an_m = _farm_stats(farms[me], day)
        rd_o, gr_o, an_o = _farm_stats(farms[1 - me], day)
        rate = shop_rate(shops)
        hf = np.array(self.hist_flow[-TPD:]) if self.hist_flow else np.zeros((1, len(PRODUCTS)))
        remaining = (EPISODE - 1 - step) / EPISODE
        P = np.zeros((len(PRODUCTS), PF), dtype=np.float32)
        for k, p in enumerate(PRODUCTS):
            b, T = BASE[p], TPARAM[p]
            iv = inv[p]
            pr = prices[p]
            f = P[k]
            f[0] = pr / b
            f[1] = (iv - I0) / T
            f[2] = math.log1p(stock[k]) / 5
            f[3] = math.log1p(shed.get(p, 0)) / 5
            f[4] = math.log1p(min(base_q[k], 999)) / 7
            f[5] = min(base_q[k], 999) / max(1.0, stock[k]) if stock[k] > 0 else 0.0
            for j, n in enumerate((1, 5, 20, 60)):        # marginal price impact of selling n units
                f[6 + j] = market_price(p, iv + n) / b
            f[10] = market_price(p, iv - 24) / b              # price after one more day of scarcity
            f[11] = math.log1p(rd_m[k]) / 4
            f[12] = math.log1p(gr_m[k]) / 4
            f[13] = math.log1p(an_m[k]) / 3
            f[14] = math.log1p(rd_o[k]) / 4
            f[15] = math.log1p(gr_o[k]) / 4
            f[16] = math.log1p(an_o[k]) / 3
            f[17] = rate[k] / 24
            f[18] = math.copysign(math.log1p(abs(hf[-1, k])), hf[-1, k]) / 4
            f[19] = math.copysign(math.log1p(abs(hf[:, k].sum())), hf[:, k].sum()) / 5
            f[20] = (pr - self.day_start_price[k]) / b if self.day_start_price is not None else 0.0
            f[21] = remaining
            f[22] = 1.0 if b > 100 else 0.0
            f[23] = stock[k] * pr / 1e4
            f[24] = self.my_last_orders[k] / 50
            f[25] = min(1.0, stock[k] / max(1.0, rate[k]))     # days of town demand held
            f[26 + (k % 6)] = 1.0                               # cheap positional hint (+ learned emb in net)
        mm, mo = farms[me]["money"], farms[1 - me]["money"]
        G = np.zeros(GF, dtype=np.float32)
        G[0] = day / 30
        G[1] = (step % TPD) / TPD
        G[2] = remaining
        G[3] = mm / 1e5
        G[4] = mo / 1e5
        G[5] = (mm - mo) / 1e4
        G[6] = sum(v for v in shed.values()) / 100
        G[7] = len(shops) / 8
        G[8] = len(farms[me]["unlocked_quadrants"]) / 4
        G[9] = len(farms[1 - me]["unlocked_quadrants"]) / 4
        G[10] = len(farms[me]["hands"]) / 12
        G[11] = len(farms[1 - me]["hands"]) / 12
        G[12] = 1.0 if step >= EPISODE - 48 else 0.0
        G[13] = 1.0 if step >= EPISODE - 12 else 0.0
        G[14] = math.sin(2 * math.pi * (step % TPD) / TPD)
        G[15] = math.cos(2 * math.pi * (step % TPD) / TPD)
        return P, G, stock, base_q

    def pop_day_targets(self):
        out, self.day_targets = self.day_targets, []
        return out


def bins_to_orders(bins, stock, keep_market, base_market):
    """Turn per-product sell bins into market orders, keeping the base agent's non-SELL orders."""
    orders = [o for o in (base_market or []) if not (isinstance(o, list) and o and o[0] == "SELL")]
    if all(int(b) == BASE_BIN for b in bins):
        return list(base_market or [])
    sells = []
    for k, b in enumerate(bins):
        spec = SELL_BINS[int(b)]
        if spec == "BASE":
            sells += [o for o in (base_market or []) if isinstance(o, list) and len(o) >= 3
                      and o[0] == "SELL" and o[1] == PRODUCTS[k]]
            continue
        if spec == 0.0:
            continue
        if spec == "ONE":
            q = 1
        elif spec == 1.0:
            q = 999
        else:
            q = int(math.ceil(spec * stock[k]))
        if q > 0 and stock[k] > 0:
            sells.append(["SELL", PRODUCTS[k], q])
    # Base agents put sells first (sell before buying feed); keep that convention, respect the 10-order cap.
    return (sells + orders)[:10] if keep_market else sells[:10]


def label_bins(market, stock):
    """Map an observed market order list to per-product sell bins (for behaviour cloning)."""
    q = np.zeros(len(PRODUCTS))
    for o in (market or []):
        if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in PIDX:
            try:
                q[PIDX[o[1]]] += float(o[2])
            except (TypeError, ValueError):
                pass
    out = np.zeros(len(PRODUCTS), dtype=np.int64)
    for k in range(len(PRODUCTS)):
        if q[k] <= 0:
            out[k] = 0
        elif stock[k] <= 0 or q[k] >= stock[k]:
            out[k] = 4
        elif q[k] <= 1:
            out[k] = 1
        else:
            fr = q[k] / stock[k]
            out[k] = 2 if fr <= 0.375 else (3 if fr <= 0.75 else 4)
    return out
