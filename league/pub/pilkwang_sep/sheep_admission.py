"""Explicit sheep-investment stress scenarios; not a profit bound or simulator.

Animal dawn arithmetic follows the pinned Kaggriculture 1.32.7 source. Future
service, placement, sales and rival adoption are declared assumptions. Only
current public farm/market observations and the parent's own hire plan enter.
"""

from __future__ import annotations

from copy import deepcopy
import math

try:
    from .market_primitives import price_at
except ImportError:  # Standalone submission directory.
    from market_primitives import price_at

ENGINE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
DEFAULT_SCENARIOS = {
    "project_sheep": 6,
    "placement_delay_days": 1,
    "project_sale_hour": 20,
    "delivery_fraction": 0.75,
    "minimum_scenario_surplus": 0,
    "scenarios": [
        {"name": "visible_herds_frozen_shops", "rival_adoption_sheep": 0,
         "rival_adoption_delay_days": 3, "feed_cost_multiplier": 1.0},
        {"name": "six_sheep_adoption_expensive_feed", "rival_adoption_sheep": 6,
         "rival_adoption_delay_days": 3, "feed_cost_multiplier": 1.25},
    ],
}


def _integer(value, name, minimum=0, maximum=1000000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def fib(index):
    _integer(index, "hire index", maximum=64)
    a, b = 1, 1
    for _ in range(index):
        a, b = b, a + b
    return a


def advance_sheep(tile, next_day, *, fed=True, cared=True):
    """One sheep dawn transition, matching engine ordering and capped storage."""
    result = deepcopy(tile)
    result["consecutive_unfed"] = 0 if fed else result["consecutive_unfed"] + 1
    if result["consecutive_unfed"] >= 2:
        return None
    age = next_day - result["placed_day"] - 6
    if age >= 0 and age % 3 == 0:
        bonus = result.get("pending_care_bonus", 0) if fed else 0
        result["yield_units"] = min(6, result["yield_units"] + 1 + bonus)
        result["pending_care_bonus"] = 0
    if cared and fed:
        result["pending_care_bonus"] = result.get("pending_care_bonus", 0) + 1
    result.update(fed_today=False, cared_today=False, fertilizer_available=True)
    return result


def _new_sheep(day):
    return {"animal": "SHEEP", "placed_day": day, "yield_units": 0,
            "pending_care_bonus": 0, "consecutive_unfed": 0,
            "fed_today": False, "cared_today": False}


def _sheep(farm):
    animals = [tile for row in farm["tiles"] for tile in row
               if isinstance(tile, dict) and tile.get("animal")]
    sheep = []
    for tile in animals:
        if tile["animal"] not in ("SHEEP", "COW", "GOOSE"):
            raise ValueError("unknown animal")
        if tile["animal"] != "SHEEP":
            continue
        for key in ("placed_day", "yield_units", "pending_care_bonus", "consecutive_unfed"):
            _integer(tile[key], key, maximum=1000)
        for key in ("fed_today", "cared_today"):
            if type(tile[key]) is not bool:
                raise ValueError(f"unknown {key}")
        if tile["yield_units"] > 6:
            raise ValueError("invalid sheep yield")
        sheep.append(deepcopy(tile))
    return sheep, len(animals)


def _sell(inventory, quantity, curve):
    proceeds = 0
    for _ in range(quantity):
        quote = price_at(inventory, curve)
        proceeds += quote
        if quote > 1:
            inventory += 1
    return proceeds, inventory


def _buy(inventory, quantity, curve):
    cost = 0
    for _ in range(quantity):
        cost += price_at(inventory - 1, curve)
        inventory -= 1
    return cost, inventory


def _projection(observation, params, config, scenario, with_project):
    """Deterministic partial-market scenario, never a reachable world claim."""
    start = observation["step"]
    start_day = start // 24
    own, rival = [observation["farms"][i] for i in
                  (observation["player"], 1 - observation["player"])]
    own_sheep, own_feed = _sheep(own)
    rival_sheep, rival_feed = _sheep(rival)
    project, adopters = [], []
    depth = {i: observation["market"]["inventory"][i] for i in ("WOOL", "WHEAT")}
    revenues = {"existing_wool": 0, "project_wool": 0}
    quantities = {"existing_wool": 0, "rival_wool": 0, "project_wool": 0,
                  "project_produced_wool": 0, "project_feed_purchased": 0,
                  "project_animal_feed_obligations": 0}
    own_feed_cost = direct_project_feed_cost = 0.0
    daily = []
    for step in range(start, 719):
        day, hour = divmod(step, 24)
        if with_project and day == start_day + config["placement_delay_days"] and hour == 0:
            project = [_new_sheep(day) for _ in range(config["project_sheep"])]
        if day == start_day + scenario["rival_adoption_delay_days"] and hour == 0:
            adopters = [_new_sheep(day) for _ in range(scenario["rival_adoption_sheep"])]
        if hour == 0 or step == start:
            # Daily gross buying is an explicit stress: no hidden stocks or
            # unmodeled crop harvest is credited. Rival buys precede ours.
            _, depth["WHEAT"] = _buy(depth["WHEAT"], rival_feed + len(adopters), params["WHEAT"])
            background, depth["WHEAT"] = _buy(depth["WHEAT"], own_feed, params["WHEAT"])
            project_units = config["project_sheep"] if with_project else 0
            purchase, depth["WHEAT"] = _buy(depth["WHEAT"], project_units, params["WHEAT"])
            multiplier = scenario["feed_cost_multiplier"]
            own_feed_cost += multiplier * (background + purchase)
            direct_project_feed_cost += multiplier * purchase
            quantities["project_feed_purchased"] += project_units
            quantities["project_animal_feed_obligations"] += len(project)
        if hour == 18:
            rival_units = sum(t["yield_units"] for t in rival_sheep + adopters)
            _, depth["WOOL"] = _sell(depth["WOOL"], rival_units, params["WOOL"])
            quantities["rival_wool"] += rival_units
            for tile in rival_sheep + adopters:
                tile["yield_units"] = 0
            own_units = sum(t["yield_units"] for t in own_sheep)
            cash, depth["WOOL"] = _sell(depth["WOOL"], own_units, params["WOOL"])
            revenues["existing_wool"] += cash
            quantities["existing_wool"] += own_units
            for tile in own_sheep:
                tile["yield_units"] = 0
        if hour == config["project_sale_hour"]:
            produced = sum(t["yield_units"] for t in project)
            delivered = math.floor(produced * config["delivery_fraction"])
            cash, depth["WOOL"] = _sell(depth["WOOL"], delivered, params["WOOL"])
            revenues["project_wool"] += cash
            quantities["project_wool"] += delivered
            quantities["project_produced_wool"] += produced
            for tile in project:
                tile["yield_units"] = 0
        if step % 4 == 0:
            for shop in observation["town"]["unlocked_shops"]:
                products = SHOPS[shop]
                for item in depth:
                    if item in products:
                        depth[item] -= 2 if len(products) == 1 else 1
        if step % 24 == 0:
            for item in depth:
                depth[item] -= 1
        if hour == 23:
            # Perfect future service is a declared scenario, not an action
            # reachability assertion; delivery haircut is charged separately.
            own_sheep = [advance_sheep(t, day + 1) for t in own_sheep]
            rival_sheep = [advance_sheep(t, day + 1) for t in rival_sheep]
            project = [advance_sheep(t, day + 1) for t in project]
            adopters = [advance_sheep(t, day + 1) for t in adopters]
            daily.append({"day": day, "inventory": dict(depth)})
    return {"revenues": revenues, "quantities": quantities,
            "own_feed_cost": own_feed_cost,
            "direct_project_feed_cost": direct_project_feed_cost,
            "end_inventory": depth, "daily_market": daily}


def evaluate_admission(observation, native_hires_by_day, *, params, scenario_config=None):
    """Return ALLOW/DECLINE or UNKNOWN (caller preserves the parent on UNKNOWN).

    This is a policy's explicit scenario test, not a guaranteed financial
    bound, expected future profit, current-meta estimate or causal win estimate.
    """
    fallback = {"status": "UNKNOWN", "decision": "PARENT_FALLBACK",
                "model_kind": "declared_partial_market_scenarios", "reason": None}
    try:
        config = deepcopy(DEFAULT_SCENARIOS if scenario_config is None else scenario_config)
        step = _integer(observation["step"], "step", maximum=718)
        player = _integer(observation["player"], "player", maximum=1)
        if step not in (288, 289):
            raise ValueError("outside initial day12 admission window")
        if len(observation["farms"]) != 2:
            raise ValueError("expected two public farms")
        for farm in observation["farms"]:
            if len(farm["tiles"]) != 10 or any(len(row) != 10 for row in farm["tiles"]):
                raise ValueError("unsupported board")
            _sheep(farm)
        for item in ("WHEAT", "WOOL"):
            depth = _integer(observation["market"]["inventory"][item], item, minimum=-1000000)
            if observation["market"]["prices"][item] != price_at(depth, params[item]):
                raise ValueError(f"quote mismatch: {item}")
        shops = observation["town"]["unlocked_shops"]
        if not isinstance(shops, list) or len(shops) > 8 or any(s not in SHOPS for s in shops):
            raise ValueError("unknown shops")
        if config["project_sheep"] != 6:
            raise ValueError("parent project must remain six sheep")
        _integer(config["placement_delay_days"], "placement delay", 0, 10)
        _integer(config["project_sale_hour"], "sale hour", 19, 22)
        if not 0 < config["delivery_fraction"] <= 1:
            raise ValueError("invalid delivery fraction")
        if not math.isfinite(config["minimum_scenario_surplus"]):
            raise ValueError("invalid scenario threshold")
        if len(config["scenarios"]) != 2:
            raise ValueError("exactly two declared scenarios required")
        for scenario in config["scenarios"]:
            _integer(scenario["rival_adoption_sheep"], "adoption sheep", 0, 6)
            _integer(scenario["rival_adoption_delay_days"], "adoption delay", 1, 10)
            if not 1 <= scenario["feed_cost_multiplier"] <= 3:
                raise ValueError("invalid feed stress multiplier")
        labor_by_day = {}
        for day in range(12, 30):
            hires = _integer(native_hires_by_day[str(day)], "native daily hires", 0, 30)
            labor_by_day[str(day)] = fib(hires) + fib(hires + 1)
        labor = sum(labor_by_day.values())
        scenarios = []
        for scenario in config["scenarios"]:
            without = _projection(observation, params, config, scenario, False)
            with_project = _projection(observation, params, config, scenario, True)
            revenue_delta = sum(with_project["revenues"].values()) - sum(without["revenues"].values())
            feed_delta = with_project["own_feed_cost"] - without["own_feed_cost"]
            surplus = revenue_delta - feed_delta - 7000 - labor
            scenarios.append({"name": scenario["name"], "assumptions": scenario,
                              "with_project": with_project, "without_project": without,
                              "incremental_wool_revenue": revenue_delta,
                              "incremental_feed_cost": feed_delta,
                              "scenario_surplus": surplus})
        worst = min(s["scenario_surplus"] for s in scenarios)
        return {"status": "EVALUATED", "decision": "ALLOW" if worst >= config["minimum_scenario_surplus"] else "DECLINE",
                "reason": "all_declared_scenarios_pass" if worst >= config["minimum_scenario_surplus"] else "declared_scenario_cost_not_recovered",
                "model_kind": "declared_partial_market_scenarios", "worst_scenario_surplus": worst,
                "configuration": config, "scenarios": scenarios,
                "costs": {"land_and_sheep": 7000, "incremental_labor": labor,
                          "labor_by_day": labor_by_day, "planned_feed_units": 108,
                          "fertilizer_revenue_credit": 0, "delivery_and_service_haircut": 1 - config["delivery_fraction"]},
                "observables": {"step": step, "player": player, "known_shops": list(shops),
                                "own_sheep": _sheep(observation["farms"][player])[0],
                                "rival_sheep": _sheep(observation["farms"][1-player])[0],
                                "market_inventory": {i: observation["market"]["inventory"][i] for i in ("WOOL", "WHEAT")}},
                "limitations": ["No future shops or hidden shed contents are inferred.",
                    "Known shops repeat; future shop additions are ignored.",
                    "All observed animals and hypothetical adopters receive daily gross market feed; crop supply is unmodeled.",
                    "Public sheep receive ideal daily care; production follows age and pending bonus, delivery loses the declared fraction.",
                    "Existing rival wool sells before existing own wool, followed by new-project delivery.",
                    "Only wool and feed market effects are modeled; crop, other animal products, land alternatives and action reachability are unpriced.",
                    "The model is a fixed stress policy, not a profit lower bound or strength estimate."]}
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        fallback["reason"] = str(error)
        return fallback
