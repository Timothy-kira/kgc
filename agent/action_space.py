"""Closed candidate action space for CLM-style action heads (replaces open-ended token decoding).

Following Stanford/Hazy Research CLM (Contrastive-LM/CLM, `heads.py`): a state head and an action head
project into a shared space and a candidate is scored by  exp(logit_scale) * cos(state_head(h), action_head(a)).
Here every step's action is decomposed into decision *slots*, each choosing one candidate from a closed set:

    <ACT> | farmer slot | hand slot x n_hands | market slot x k | STOP
unit candidates   : every unit action (moves, tile ops, PLANT crop, PICKUP/PLACE item count, <NONE>)
market candidates : op x item x quantity (exact 1..99 + large buckets), HIRE, BUY_LAND, <EMPTY> (no-op order
                    that only occupies an index), STOP (end of the order list)
Candidates are described by (kind, op, item, qty) ids -> embeddings -> action head, so unseen combinations
still get sensible embeddings and the projected candidate matrix can be cached at inference.
All candidates are valid actions, so decoding can only produce legal actions.
"""
import numpy as np

UNIT_OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP", "PICKUP", "PLACE", "PLANT", "<NONE>"]
MARKET_OPS = ["SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND", "<EMPTY>", "<STOP>"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW",
         "SHEEP"]
CROPS = ITEMS[:5]
UNIT_COUNTS = list(range(1, 21)) + [99]                    # PICKUP / PLACE counts (+ "default" = no count)
ORDER_QTYS = list(range(1, 100)) + [100, 200, 500, 999, 1000, 9999]

# description vocab (shared by the action encoder)
OP_VOCAB = ["U:" + o for o in UNIT_OPS] + ["M:" + o for o in MARKET_OPS]
OPI = {o: i for i, o in enumerate(OP_VOCAB)}
ITEM_VOCAB = ["<none>"] + ITEMS
ITI = {o: i for i, o in enumerate(ITEM_VOCAB)}
QTY_VOCAB = ["<none>", "<default>"] + [str(q) for q in sorted(set(UNIT_COUNTS + ORDER_QTYS))]
QTI = {o: i for i, o in enumerate(QTY_VOCAB)}


def _build():
    unit, market = [], []
    for op in UNIT_OPS:
        if op == "PLANT":
            unit += [("PLANT", c, None) for c in CROPS] + [("PLANT", None, None)]
        elif op in ("PICKUP", "PLACE"):
            unit.append((op, None, None))
            for it in ITEMS:
                unit.append((op, it, "default"))
                unit += [(op, it, n) for n in UNIT_COUNTS]
        else:
            unit.append((op, None, None))
    for op in MARKET_OPS:
        if op in ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"):
            market += [(op, it, q) for it in ITEMS for q in ORDER_QTYS]
        else:
            market.append((op, None, None))
    return unit, market


UNIT_CANDS, MARKET_CANDS = _build()
N_UNIT, N_MARKET = len(UNIT_CANDS), len(MARKET_CANDS)
UNIT_IDX = {c: i for i, c in enumerate(UNIT_CANDS)}
MARKET_IDX = {c: i for i, c in enumerate(MARKET_CANDS)}
STOP = MARKET_IDX[("<STOP>", None, None)]
EMPTY = MARKET_IDX[("<EMPTY>", None, None)]
UNIT_NONE = UNIT_IDX[("<NONE>", None, None)]
MAX_ORDERS = 10

SLOT_FARMER, SLOT_HAND, SLOT_MARKET = 0, 1, 2


def _desc(kind, c):
    op, it, q = c
    o = OPI[("U:" if kind == 0 else "M:") + op]
    i = ITI[it] if it is not None else 0
    if q is None:
        qi = 0
    elif q == "default":
        qi = 1
    else:
        qi = QTI[str(q)]
    return (kind, o, i, qi)


# [N, 4] description ids for the action encoder
UNIT_DESC = np.array([_desc(0, c) for c in UNIT_CANDS], np.int64)
MARKET_DESC = np.array([_desc(1, c) for c in MARKET_CANDS], np.int64)


def _nearest(v, choices):
    return min(choices, key=lambda c: abs(c - v))


def unit_to_cand(u):
    if not isinstance(u, list) or not u or u[0] not in UNIT_OPS:
        return UNIT_NONE
    op = u[0]
    if op == "PLANT":
        return UNIT_IDX.get(("PLANT", u[1] if len(u) > 1 and u[1] in CROPS else None, None), UNIT_IDX[("PLANT", None, None)])
    if op in ("PICKUP", "PLACE"):
        if len(u) < 2 or u[1] not in ITEMS:
            return UNIT_IDX[(op, None, None)]
        if len(u) < 3:
            return UNIT_IDX[(op, u[1], "default")]
        try:
            n = int(u[2])
        except (TypeError, ValueError):
            return UNIT_IDX[(op, u[1], "default")]
        return UNIT_IDX[(op, u[1], n if n in UNIT_COUNTS else _nearest(n, UNIT_COUNTS))]
    return UNIT_IDX[(op, None, None)]


def order_to_cand(o):
    if not isinstance(o, list) or not o or o[0] not in MARKET_OPS[:6]:
        return EMPTY
    op = o[0]
    if op in ("HIRE", "BUY_LAND"):
        return MARKET_IDX[(op, None, None)]
    if len(o) < 3 or o[1] not in ITEMS:
        return EMPTY
    try:
        q = int(o[2])
    except (TypeError, ValueError):
        return EMPTY
    if q <= 0:
        return EMPTY
    return MARKET_IDX[(op, o[1], q if q in ORDER_QTYS else _nearest(q, ORDER_QTYS))]


def cand_to_unit(i):
    op, it, q = UNIT_CANDS[i]
    if op == "<NONE>":
        return None
    if it is None:
        return [op]
    if q is None or q == "default":
        return [op, it]
    return [op, it, q]


def cand_to_order(i):
    op, it, q = MARKET_CANDS[i]
    if op == "<EMPTY>":
        return []
    if it is None:
        return [op]
    return [op, it, q]


def action_to_decisions(action, n_hands):
    """-> list of (slot_type, candidate index). Hands are padded/truncated to the hands that exist
    (the environment ignores extra hand actions; a missing one is a no-op)."""
    a = action if isinstance(action, dict) else {}
    dec = [(SLOT_FARMER, unit_to_cand(a.get("farmer")))]
    hands = a.get("hands") if isinstance(a.get("hands"), list) else []
    for h in range(n_hands):
        dec.append((SLOT_HAND, unit_to_cand(hands[h]) if h < len(hands) else UNIT_NONE))
    market = a.get("market") if isinstance(a.get("market"), list) else []
    for o in market[:MAX_ORDERS]:
        dec.append((SLOT_MARKET, order_to_cand(o)))
    if len(market[:MAX_ORDERS]) < MAX_ORDERS:
        dec.append((SLOT_MARKET, STOP))
    return dec


def decisions_to_action(dec):
    act = {"farmer": None, "hands": [], "market": []}
    for slot, c in dec:
        if slot == SLOT_FARMER:
            act["farmer"] = cand_to_unit(c)
        elif slot == SLOT_HAND:
            act["hands"].append(cand_to_unit(c))
        elif c != STOP:
            act["market"].append(cand_to_order(c))
    if act["farmer"] is None:
        act["farmer"] = ["PASS"]
    act["hands"] = [h if h is not None else ["PASS"] for h in act["hands"]]
    return act


class SlotPlan:
    """Deterministic slot sequence for one step (the number of hands is known from the observation);
    the market part ends at STOP or after MAX_ORDERS orders."""

    def __init__(self, n_hands):
        self.n_hands = n_hands
        self.i = 0
        self.orders = 0
        self.done = False

    def next_slot(self):
        if self.i == 0:
            return SLOT_FARMER
        if self.i <= self.n_hands:
            return SLOT_HAND
        return SLOT_MARKET

    def mask(self):
        """Allowed candidates for the next slot (unit slot -> all units; market slot -> all orders, STOP
        forced once MAX_ORDERS orders were placed)."""
        s = self.next_slot()
        if s != SLOT_MARKET:
            return s, None
        if self.orders >= MAX_ORDERS:
            m = np.zeros(N_MARKET, bool)
            m[STOP] = True
            return s, m
        return s, None

    def feed(self, slot, cand):
        self.i += 1
        if slot == SLOT_MARKET:
            if cand == STOP:
                self.done = True
            else:
                self.orders += 1
                if self.orders >= MAX_ORDERS:
                    self.done = True
