"""Pure market mechanics and explicit confirmed-fill accounting; not an agent.

Pricing/sale semantics follow kaggle-environments 1.32.7. The caller supplies
resolved price parameters and post-worker shed stock. No opponent state is
inferred, and no future BUY/PICKUP, cash reserve or production guard is supplied.
Public sale-window/depth ideas are independently implemented; provenance and
the bounded differential checks are in the strategy_factory market notes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import math

PRODUCTS = frozenset("WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER".split())
_SHAPES = frozenset(("linear", "sq", "sqrt", "log", "log10", "hinge"))


def _integer(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _mapping(value, label, *, nonnegative=False):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    for key, number in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        _integer(number, label, 0 if nonnegative else None)


def _shape(name, x, throughput):
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log(1.0 + x)
    if name == "log10":
        return math.log10(1.0 + x)
    u = x / throughput
    return u + 8.0 * max(0.0, u - 1.0) ** 2


def price_at(inventory: int, curve: Mapping) -> int:
    """Price before a SELL unit, including hinge, Python rounding and floor1.

    ``curve`` is one fully resolved marketParams entry, never sparse overrides.
    Unlike engine fallback parsing, malformed or unknown curve shapes reject.
    """
    _integer(inventory, "inventory")
    if not isinstance(curve, Mapping):
        raise ValueError("curve must be a resolved mapping")
    for key in ("base", "I0", "T", "below_target", "above_target"):
        value = curve.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"curve.{key} must be finite")
    if curve["base"] <= 0 or curve["T"] <= 0 or min(curve["below_target"], curve["above_target"]) < 0:
        raise ValueError("curve base/T must be positive and amplitudes nonnegative")
    if any(curve.get(k) not in _SHAPES for k in ("below_func", "above_func")):
        raise ValueError("unknown price curve shape")
    below = inventory < curve["I0"]
    side = "below" if below else "above"
    shape = curve[f"{side}_func"]
    amplitude = curve[f"{side}_target"] * curve["base"] / _shape(shape, curve["T"], curve["T"])
    delta = curve["I0"] - inventory if below else inventory - curve["I0"]
    price = curve["base"] + (1 if below else -1) * amplitude * _shape(shape, delta, curve["T"])
    return max(1, int(round(price)))


def _sale(inventory, quantity, curve):
    proceeds = 0
    for _ in range(quantity):
        quote = price_at(inventory, curve)
        proceeds += quote
        # The engine burns price-floor sales rather than adding market supply.
        if quote > 1:
            inventory += 1
    return proceeds, inventory


@dataclass(frozen=True)
class SaleFill:
    order_index: int
    item: str
    requested: int
    filled: int
    proceeds: int


@dataclass(frozen=True)
class SaleBatch:
    accepted: bool
    reason: str
    fills: tuple[SaleFill, ...]
    proceeds: int
    inventory: dict
    shed: dict
    fees: int = 0  # No trading fee in the pinned engine's SELL commit path.


def evaluate_sales(orders, inventory, shed, params, *, quotes=None, max_units=1000) -> SaleBatch:
    """Exact own SELL-only sequence with no interleaved rival transactions.

    This is a conditional mechanical result, not a prediction of a live fill.
    Multiple orders of one item consume the updated stock/depth sequentially.
    Unknown product/params, non-SELL slots or >10 slots abstain unchanged.
    Numeric/type corruption raises ValueError; missing information abstains.
    """
    _mapping(inventory, "inventory")
    _mapping(shed, "shed", nonnegative=True)
    if quotes is not None:
        _mapping(quotes, "quotes", nonnegative=True)
    if not isinstance(orders, (list, tuple)) or not isinstance(params, Mapping):
        raise ValueError("orders must be a sequence and params a mapping")
    _integer(max_units, "max_units", 1)
    rejected = lambda reason: SaleBatch(False, reason, (), 0, dict(inventory), dict(shed))
    if len(orders) > 10:
        return rejected("order_limit")
    for order in orders:
        if not isinstance(order, (list, tuple)) or len(order) != 3 or order[0] != "SELL":
            return rejected("unsupported_order")
        _, item, quantity = order
        _integer(quantity, "sale quantity", 0)
        if not isinstance(item, str) or item not in PRODUCTS:
            return rejected("unknown_item")
        if item not in inventory or item not in params or item not in shed:
            return rejected("missing_state")
        expected = price_at(inventory[item], params[item])
        if quotes is not None and (item not in quotes or quotes[item] != expected):
            return rejected("quote_mismatch")
    available = dict(shed)
    depth = dict(inventory)
    fills = []
    total = units = 0
    for index, (_, item, requested) in enumerate(orders):
        filled = min(requested, available[item])
        units += filled
        if units > max_units:
            return rejected("unit_budget")
        proceeds, depth[item] = _sale(depth[item], filled, params[item])
        available[item] -= filled
        total += proceeds
        fills.append(SaleFill(index, item, requested, filled, proceeds))
    return SaleBatch(True, "sale_only_no_rival", tuple(fills), total, depth, available)


@dataclass(frozen=True)
class SalePriority:
    accepted: bool
    reason: str
    orders: list
    scores: tuple[tuple[str, int], ...] = ()


def prioritize_sales(orders, inventory, shed, params, *, quotes, rival_lead=None) -> SalePriority:
    """Stable impact ranking under an explicit hypothetical rival lead sale.

    Each score is exact own proceeds now minus proceeds after ``rival_lead``
    units of that product sell first. This is not simultaneous lockstep and
    is not an optimal order proof. Unknown lead or mixed/duplicate orders keep
    the caller's original slots; no sale is moved across a purchase.
    """
    baseline = evaluate_sales(orders, inventory, shed, params, quotes=quotes)
    unchanged = lambda reason: SalePriority(False, reason, deepcopy(list(orders)))
    if not baseline.accepted:
        return unchanged(baseline.reason)
    if quotes is None:
        return unchanged("unknown_quotes")
    if len({fill.item for fill in baseline.fills}) != len(baseline.fills):
        return unchanged("duplicate_product")
    if rival_lead is None:
        return unchanged("unknown_rival_lead")
    _mapping(rival_lead, "rival_lead", nonnegative=True)
    scores = []
    for fill in baseline.fills:
        if fill.item not in rival_lead:
            return unchanged("unknown_rival_lead")
        if rival_lead[fill.item] > 1000:
            return unchanged("unit_budget")
        _, delayed_inventory = _sale(inventory[fill.item], rival_lead[fill.item], params[fill.item])
        delayed, _ = _sale(delayed_inventory, fill.filled, params[fill.item])
        scores.append((fill.item, fill.proceeds - delayed))
    by_item = dict(scores)
    result = sorted(deepcopy(list(orders)), key=lambda order: -by_item[order[1]])
    return SalePriority(True, "explicit_lead_sale_stress", result, tuple(scores))


@dataclass(frozen=True)
class AdvanceRequest:
    request_id: str
    created_step: int
    item: str
    allocations: tuple[tuple[int, int], ...]
    confirmed_fill: int | None = None
    evidence_ref: str | None = None

    def __post_init__(self):
        if (not isinstance(self.request_id, str) or not self.request_id
                or not isinstance(self.item, str) or self.item not in PRODUCTS):
            raise ValueError("request identity/item is invalid")
        _integer(self.created_step, "created_step", 0)
        if not isinstance(self.allocations, tuple) or not self.allocations:
            raise ValueError("allocations must be a nonempty tuple")
        last = self.created_step
        for allocation in self.allocations:
            if not isinstance(allocation, tuple) or len(allocation) != 2:
                raise ValueError("each allocation must be a (due_step, quantity) tuple")
            due, quantity = allocation
            _integer(due, "due_step", last + 1)
            _integer(quantity, "allocation quantity", 1)
            last = due
        if self.confirmed_fill is not None:
            _integer(self.confirmed_fill, "confirmed_fill", 0)
            if self.confirmed_fill > sum(q for _, q in self.allocations):
                raise ValueError("confirmed fill exceeds requested quantity")
            if not isinstance(self.evidence_ref, str) or not self.evidence_ref:
                raise ValueError("confirmed fill needs an evidence_ref")
        elif self.evidence_ref is not None:
            raise ValueError("unconfirmed request cannot carry confirmation evidence")


@dataclass(frozen=True)
class SaleLedger:
    """One caller-owned game namespace. Start a new instance at step zero.

    The key is a local label, not an external Kaggle episode identifier. No
    hidden episode ID or seed is required by a runtime integration.

    Requests retain confirmation identities for idempotence. Only confirmed
    quantities appear in debts; unpaid due entries never silently expire.
    """
    game_key: str
    requests: tuple[AdvanceRequest, ...] = ()
    debts: tuple[tuple[int, str, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.game_key, str) or not self.game_key:
            raise ValueError("game_key must be nonempty")
        if not isinstance(self.requests, tuple) or not all(isinstance(r, AdvanceRequest) for r in self.requests):
            raise ValueError("requests must be a tuple of AdvanceRequest")
        if len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError("duplicate request identity")
        if not isinstance(self.debts, tuple):
            raise ValueError("debts must be a tuple")
        confirmed = {}
        for request in self.requests:
            remaining = request.confirmed_fill or 0
            for due, quantity in request.allocations:
                take = min(remaining, quantity)
                key = (due, request.item)
                confirmed[key] = confirmed.get(key, 0) + take
                remaining -= take
        keys = []
        for debt in self.debts:
            if not isinstance(debt, tuple) or len(debt) != 3:
                raise ValueError("each debt must be a (due_step, item, quantity) tuple")
            due, item, quantity = debt
            _integer(due, "due_step", 0)
            _integer(quantity, "debt quantity", 1)
            if not isinstance(item, str) or item not in PRODUCTS:
                raise ValueError("unknown debt item")
            if quantity > confirmed.get((due, item), 0):
                raise ValueError("debt exceeds confirmed fills")
            keys.append((due, item))
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate debt key")


def reserve_advance(ledger, request_id, step, item, available, due_sales, *, obligations_cleared=False):
    """Reserve known future SELL units, without yet creating repayment debt.

    ``available`` is the caller's safe uncommitted post-worker stock, not gross
    shed contents. ``due_sales`` gives total planned SELL units at each future
    step. The caller must check BUY/PICKUP, route boundaries, existing sales and
    production/capital obligations. Default false deliberately makes no change.
    """
    if not isinstance(ledger, SaleLedger):
        raise ValueError("ledger must be SaleLedger")
    _integer(step, "step", 0)
    _integer(available, "available", 0)
    if (not isinstance(request_id, str) or not request_id
            or not isinstance(item, str) or item not in PRODUCTS):
        raise ValueError("invalid request_id/item")
    if not isinstance(obligations_cleared, bool) or not isinstance(due_sales, Mapping):
        raise ValueError("obligations_cleared must be bool; due_sales must be mapping")
    for due, quantity in due_sales.items():
        _integer(due, "due_step", step + 1)
        _integer(quantity, "due quantity", 0)
    if any(r.request_id == request_id for r in ledger.requests):
        raise ValueError("duplicate request identity")
    if not obligations_cleared or not available:
        return ledger
    used = {due: quantity for due, name, quantity in ledger.debts if name == item}
    for request in ledger.requests:
        if request.item == item and request.confirmed_fill is None:
            for due, quantity in request.allocations:
                used[due] = used.get(due, 0) + quantity
    allocations = []
    for due in sorted(due_sales):
        take = min(available, max(0, due_sales[due] - used.get(due, 0)))
        if take:
            allocations.append((due, take))
            available -= take
    if not allocations:
        return ledger
    request = AdvanceRequest(request_id, step, item, tuple(allocations))
    return replace(ledger, requests=(*ledger.requests, request))


def confirm_advance(ledger, request_id, filled_quantity, *, evidence_ref=None):
    """Accept an actual attributed fill receipt; None means unknown, not zero.

    This function does not infer fills from requested actions or net shed deltas.
    A partial fill is assigned to earliest due units first. A receipt is final
    for one request, rather than an incremental quantity. Repeating it is safe.
    """
    if not isinstance(ledger, SaleLedger):
        raise ValueError("ledger must be SaleLedger")
    index = next((i for i, r in enumerate(ledger.requests) if r.request_id == request_id), None)
    if index is None:
        raise ValueError("unknown request identity")
    if filled_quantity is None:
        return ledger
    request = ledger.requests[index]
    confirmed = replace(request, confirmed_fill=filled_quantity, evidence_ref=evidence_ref)
    if request.confirmed_fill is not None:
        if request.confirmed_fill != filled_quantity:
            raise ValueError("conflicting confirmation")
        return ledger
    debts = {(due, item): quantity for due, item, quantity in ledger.debts}
    remaining = filled_quantity
    for due, quantity in request.allocations:
        take = min(quantity, remaining)
        if take:
            key = (due, request.item)
            debts[key] = debts.get(key, 0) + take
            remaining -= take
    requests = list(ledger.requests)
    requests[index] = confirmed
    return replace(ledger, requests=tuple(requests), debts=tuple((d, i, q) for (d, i), q in sorted(debts.items())))


def repay_due_sales(ledger, step, orders):
    """Subtract only this due step/item's confirmed fill; preserve zero slots.

    Unmatched/partially repaid debts remain visible for explicit reconciliation.
    Later steps do not automatically consume overdue or future debt.
    This is a pure transaction proposal: evaluate retries/alternative actions
    from the SAME input ledger. Commit the returned ledger exactly once, only
    together with the final emitted action, and cache that action for retries.
    Reusing the returned ledger with the original orders is not a retry. A
    confirmation arriving after its due step stays as overdue debt; integration
    must reconcile it explicitly rather than assume a later sale repays it.
    """
    if not isinstance(ledger, SaleLedger) or not isinstance(orders, (list, tuple)):
        raise ValueError("expected SaleLedger and order sequence")
    _integer(step, "step", 0)
    debts = {(d, i): q for d, i, q in ledger.debts}
    result = deepcopy(list(orders))
    for index, order in enumerate(result):
        if not isinstance(order, (list, tuple)):
            raise ValueError("each order must be a sequence")
        if not order or order[0] != "SELL":
            continue
        if len(order) != 3 or not isinstance(order[1], str):
            raise ValueError("malformed SELL order")
        _integer(order[2], "sale quantity", 0)
        key = (step, order[1])
        take = min(order[2], debts.get(key, 0))
        if take:
            result[index] = ["SELL", order[1], order[2] - take]
            debts[key] -= take
    updated = replace(ledger, debts=tuple((d, i, q) for (d, i), q in sorted(debts.items()) if q))
    return updated, result
