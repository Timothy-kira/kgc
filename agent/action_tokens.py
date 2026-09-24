"""Exact, reversible tokenisation of Kaggriculture actions for the decoder-only policy.

One step's action  {"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}
is written as
    <ACT> <FARMER> unit  <HAND> unit ... <HAND> unit  <MARKET> order ... order <EOS>
unit   := OP | OP ITEM | OP ITEM NUM | OP ITEM <DEFAULT>          (PICKUP/PLACE count optional)
order  := SELL|BUY_SEED|BUY_PRODUCT|BUY_ANIMAL ITEM NUM | HIRE | BUY_LAND | <EMPTY>
NUM    := N0..N99 | <HI> Nh Nl   (value = 100*h + l, up to 9999; larger values clamp to 9999)
<BASE> (in place of the whole action body) means "copy the base executor's action".

`Grammar` gives the set of legal next tokens after any prefix (used to mask decoding).
"""
import numpy as np

SPECIAL = ["<PAD>", "<ACT>", "<FARMER>", "<HAND>", "<MARKET>", "<EOS>", "<BASE>", "<EMPTY>", "<DEFAULT>", "<HI>",
           "<NONE>"]
UNIT_OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP", "PICKUP", "PLACE", "PLANT"]
MARKET_OPS = ["SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW",
         "SHEEP"]
NUMS = [f"N{i}" for i in range(100)]
VOCAB = SPECIAL + ["U:" + o for o in UNIT_OPS] + ["M:" + o for o in MARKET_OPS] + ["I:" + i for i in ITEMS] + NUMS
TOK = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)

UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}          # take an item argument
UNIT_COUNT_OPS = {"PICKUP", "PLACE"}                  # optional count after the item
ORDER_ARG_OPS = {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
# Market orders accept any item token: top players deliberately queue no-op orders (e.g. BUY_PRODUCT EGG,
# empty []) to shift the index at which their real orders meet the opponent's in lockstep execution.
ITEMS_FOR = {"PLANT": CROPS, "PICKUP": ITEMS, "PLACE": ITEMS, "SELL": ITEMS, "BUY_SEED": ITEMS,
             "BUY_PRODUCT": ITEMS, "BUY_ANIMAL": ITEMS}


class TokenizeError(ValueError):
    pass


def _num(n):
    n = int(n)
    if n < 0:
        raise TokenizeError(f"negative quantity {n}")
    if n < 100:
        return [TOK[f"N{n}"]]
    n = min(n, 9999)
    return [TOK["<HI>"], TOK[f"N{n // 100}"], TOK[f"N{n % 100}"]]


def _unit(u):
    if u is None:
        return [TOK["<NONE>"]]
    if not isinstance(u, list) or not u or u[0] not in UNIT_OPS:
        raise TokenizeError(f"bad unit action {u!r}")
    op = u[0]
    out = [TOK["U:" + op]]
    if op in UNIT_ITEM_OPS:
        if len(u) < 2:
            out.append(TOK["<DEFAULT>"])        # PLACE/PICKUP/PLANT with no argument (a no-op)
            return out
        if u[1] not in ITEMS:
            raise TokenizeError(f"bad item {u!r}")
        out.append(TOK["I:" + u[1]])
        if op in UNIT_COUNT_OPS:
            out += _num(u[2]) if len(u) >= 3 else [TOK["<DEFAULT>"]]
    elif len(u) > 1:
        raise TokenizeError(f"unexpected args {u!r}")
    return out


def _order(o):
    if not isinstance(o, list) or not o:
        return [TOK["<EMPTY>"]]
    op = o[0]
    if op not in MARKET_OPS:
        raise TokenizeError(f"bad order {o!r}")
    if op in ORDER_ARG_OPS:
        if len(o) < 3 or o[1] not in ITEMS:
            raise TokenizeError(f"bad order {o!r}")
        return [TOK["M:" + op], TOK["I:" + o[1]]] + _num(o[2])
    return [TOK["M:" + op]]


def encode(action):
    """Action dict -> token ids (starting with <ACT>, ending with <EOS>)."""
    if action == "BASE":
        return [TOK["<ACT>"], TOK["<BASE>"], TOK["<EOS>"]]
    if not isinstance(action, dict):
        raise TokenizeError(f"not a dict: {action!r}")
    out = [TOK["<ACT>"], TOK["<FARMER>"]] + _unit(action.get("farmer"))
    for h in action.get("hands") or []:
        out += [TOK["<HAND>"]] + _unit(h)
    out.append(TOK["<MARKET>"])
    for o in action.get("market") or []:
        out += _order(o)
    out.append(TOK["<EOS>"])
    return out


def decode(ids):
    """Token ids (one step, from <ACT> to <EOS>) -> action dict ("BASE" for the copy token)."""
    t = [VOCAB[i] for i in ids]
    if t[:2] == ["<ACT>", "<BASE>"]:
        return "BASE"
    pos = 1
    act = {"farmer": None, "hands": [], "market": []}

    def num(p):
        if t[p] == "<HI>":
            return 100 * int(t[p + 1][1:]) + int(t[p + 2][1:]), p + 3
        return int(t[p][1:]), p + 1

    def unit(p):
        if t[p] == "<NONE>":
            return None, p + 1
        op = t[p][2:]
        p += 1
        if op in UNIT_ITEM_OPS:
            if t[p] == "<DEFAULT>":
                return [op], p + 1
            item = t[p][2:]
            p += 1
            if op in UNIT_COUNT_OPS:
                if t[p] == "<DEFAULT>":
                    return [op, item], p + 1
                n, p = num(p)
                return [op, item, n], p
            return [op, item], p
        return [op], p

    assert t[pos] == "<FARMER>"
    act["farmer"], pos = unit(pos + 1)
    while t[pos] == "<HAND>":
        u, pos = unit(pos + 1)
        act["hands"].append(u)
    assert t[pos] == "<MARKET>"
    pos += 1
    while t[pos] != "<EOS>":
        if t[pos] == "<EMPTY>":
            act["market"].append([])
            pos += 1
            continue
        op = t[pos][2:]
        pos += 1
        if op in ORDER_ARG_OPS:
            item = t[pos][2:]
            n, pos = num(pos + 1)
            act["market"].append([op, item, n])
        else:
            act["market"].append([op])
    return act


# ---------------------------------------------------------------------------------------------- grammar
def _ids(names):
    m = np.zeros(V, bool)
    for n in names:
        m[TOK[n]] = True
    return m


_NUM_MASK = _ids(["<HI>"] + NUMS)
_DIGIT_MASK = _ids(NUMS)


class Grammar:
    """Incremental legal-next-token masks for one step of decoding.

    `max_hands`: number of hired hands this step (the action may list at most that many).
    `max_orders`: market order cap (10).
    """

    def __init__(self, max_hands, max_orders=10, allow_base=True):
        self.max_hands, self.max_orders, self.allow_base = max_hands, max_orders, allow_base
        self.state = "start"
        self.n_hands = 0
        self.n_orders = 0
        self.op = None
        self.num_left = 0

    def mask(self):
        s = self.state
        if s == "start":
            return _ids(["<ACT>"])
        if s == "act":
            return _ids(["<FARMER>"] + (["<BASE>"] if self.allow_base else []))
        if s == "base":
            return _ids(["<EOS>"])
        if s == "unit":
            return _ids(["U:" + o for o in UNIT_OPS] + ["<NONE>"])
        if s == "unit_item":
            return _ids(["I:" + i for i in ITEMS_FOR[self.op]] + ["<DEFAULT>"])
        if s == "unit_count":
            m = _NUM_MASK.copy()
            m[TOK["<DEFAULT>"]] = True
            return m
        if s == "after_unit":
            names = ["<MARKET>"] + (["<HAND>"] if self.n_hands < self.max_hands else [])
            return _ids(names)
        if s == "order":
            names = ["<EOS>"]
            if self.n_orders < self.max_orders:
                names += ["M:" + o for o in MARKET_OPS] + ["<EMPTY>"]
            return _ids(names)
        if s == "order_item":
            return _ids(["I:" + i for i in ITEMS_FOR[self.op]])
        if s == "order_num":
            return _NUM_MASK
        if s == "digits":
            return _DIGIT_MASK
        return _ids(["<EOS>"])

    def feed(self, tid):
        t = VOCAB[tid]
        s = self.state
        if s == "start":
            self.state = "act"
        elif s == "act":
            self.state = "base" if t == "<BASE>" else "unit"
        elif s == "base":
            self.state = "done"
        elif s == "unit":
            if t == "<NONE>":
                self.state = "after_unit"
            else:
                self.op = t[2:]
                self.state = "unit_item" if self.op in UNIT_ITEM_OPS else "after_unit"
        elif s == "unit_item":
            if t == "<DEFAULT>":
                self.state = "after_unit"
            else:
                self.state = "unit_count" if self.op in UNIT_COUNT_OPS else "after_unit"
        elif s == "unit_count":
            if t == "<HI>":
                self.state, self.num_left, self._ret = "digits", 2, "after_unit"
            else:
                self.state = "after_unit"
        elif s == "after_unit":
            if t == "<HAND>":
                self.n_hands += 1
                self.state = "unit"
            else:
                self.state = "order"
        elif s == "order":
            if t == "<EOS>":
                self.state = "done"
            elif t == "<EMPTY>":
                self.n_orders += 1
            else:
                self.op = t[2:]
                self.n_orders += 1
                self.state = "order_item" if self.op in ORDER_ARG_OPS else "order"
        elif s == "order_item":
            self.state = "order_num"
        elif s == "order_num":
            if t == "<HI>":
                self.state, self.num_left, self._ret = "digits", 2, "order"
            else:
                self.state = "order"
        elif s == "digits":
            self.num_left -= 1
            if self.num_left == 0:
                self.state = self._ret
        return self.state

    @property
    def done(self):
        return self.state == "done"
