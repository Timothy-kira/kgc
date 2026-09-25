"""TopPlanner: an independent, storage-aware farm planner implementing the top teams' macro strategy
(tools/strategy_profile.py): early land (day 6/9/10), diversified crops (tomato/carrot from mid-game), geese,
continuous demand-aware selling around day boundaries, fertilizer used on one-time crops instead of dumped,
and shed capacity (100) managed explicitly. It is a complete agent (routed expert of the H-MoE layer), written
from the interpreter's rules (kaggriculture.py):

  per step: units act first (farmer, hands), then the market, then town consumption / decay / end of day
  PLANT needs the seed already bought (buy on step t, plant on t+1); WATER once a day; new seeds must be watered
  on the planting day; one-time crops must be harvested on day planted+max_yield_day (decay starts next day);
  ongoing crops produce 4 times; animals need FEED (1 wheat from the unit's inventory) daily, CARE banks a bonus,
  COLLECT_FERTILIZER gives 1/day; shed actions need a shed-access tile; inventories are dumped into the shed at
  day end (overflow lost).
"""
import math

from agent.top_experts import MARKET_PARAMS, I0, market_price, sellable

CROPS = {  # seed, first_yield_day, max_yield_day, interval, max_yield, ongoing
    "WHEAT": (10, 2, 4, 0, 6, False), "CARROT": (20, 2, 3, 0, 4, False), "TOMATO": (50, 8, 8, 1, 4, True),
    "STRAWBERRY": (100, 10, 10, 2, 4, True), "MELON": (80, 10, 12, 0, 6, False)}
ANIMALS = {"GOOSE": (300, "COOP", "EGG"), "COW": (400, "PASTURE", "MILK"), "SHEEP": (500, "PASTURE", "WOOL")}
LAND = (1000, 2000, 4000)
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
TPD, DAYS = 24, 30
SHED_CAP = 100

# DSM / Mother-Goose profile (tile targets per phase; animals as head counts)
PROFILE = {
    "land_days": (6, 10, 40),        # sweep: the 4th quadrant (4000) did not pay back
    "zone_penalty": 4,
    "hire_div": 20.0,
    "hire_cap": 11,
    "plants_per_unit": 13,
    "water_daily": 7,                # priority of optional (non-urgent, no-gain) watering; 0 = skip (sweep: 7 best)
    "phases": [  # (from_day, crop targets, animal targets)
        # early cash engine: 12 melons on day 0 (6 x 250 each around day 10-12), wheat for feed / quick cash
        # animals are the engine: fed + cared daily they bank a bonus paid on each production
        # (cow ~1.5 milk/day, sheep ~1.3 wool/day with care), so buy them early and never skip CARE
        (0, {"MELON": 12, "WHEAT": 8}, {"COW": 2, "SHEEP": 2}),
        (3, {"MELON": 12, "WHEAT": 10, "STRAWBERRY": 6}, {"COW": 4, "SHEEP": 4, "GOOSE": 1}),
        (8, {"WHEAT": 22, "STRAWBERRY": 20, "TOMATO": 10, "CARROT": 5}, {"COW": 6, "SHEEP": 4, "GOOSE": 6}),
        (20, {"WHEAT": 25, "CARROT": 14, "TOMATO": 8, "STRAWBERRY": 10}, {"COW": 6, "SHEEP": 4, "GOOSE": 6}),
    ],
    "last_animal_day": {"GOOSE": 22, "COW": 19, "SHEEP": 17},
    "sell_hours": {22, 23, 0, 1, 2},
    "sell_ratio": 0.9,          # sell while quote >= ratio * base
    "fert_min_price": 45,
    "reserve": 150,
}


def _age_ok(crop, day):
    """Can a crop planted today still produce before the season ends?"""
    seed, first, maxd, interval, maxy, ongoing = CROPS[crop]
    need = first + (1 if ongoing else maxd - first)
    return day + need <= DAYS - 1


class TopPlanner:
    def __init__(self, profile=None):
        self.p = dict(PROFILE, **(profile or {}))
        self.assign = {}         # unit index -> (kind, target pos): committed until done / invalid (reset daily)
        self.assign_day = -1

    # ------------------------------------------------------------------ helpers
    def _parse(self, obs):
        self.me = int(obs["player"])
        f = obs["farms"][self.me]
        self.farm, self.tiles = f, f["tiles"]
        self.n = len(self.tiles)
        h = self.n // 2
        self.shed_tiles = [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
        priv = obs.get("private") or {}
        self.shed = dict(priv.get("shed") or {})
        self.seeds = dict(priv.get("seeds") or {})
        self.invs = [dict(x or {}) for x in (priv.get("inventories") or [{}])]
        self.units = [tuple(f["farmer"])] + [tuple(p) for p in f["hands"]]
        while len(self.invs) < len(self.units):
            self.invs.append({})
        self.step = int(obs["step"])
        self.day, self.hour = self.step // TPD, self.step % TPD
        self.money = f["money"]
        self.prices = obs["market"]["prices"]
        self.minv = dict(obs["market"].get("inventory") or {})
        self.shed_used = sum(self.shed.values())

    def _cells(self):
        for y in range(self.n):
            for x in range(self.n):
                yield (x, y), self.tiles[y][x]

    def _phase(self):
        cur = self.p["phases"][0]
        for ph in self.p["phases"]:
            if self.day >= ph[0]:
                cur = ph
        return cur

    @staticmethod
    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _shed_dist(self, pos):
        return min(self._dist(pos, s) for s in self.shed_tiles)

    def _move_toward(self, pos, target):
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        if dx:
            return ["EAST"] if dx > 0 else ["WEST"]
        if dy:
            return ["SOUTH"] if dy > 0 else ["NORTH"]
        return None

    # ------------------------------------------------------------------ farm census
    def _census(self):
        crops, animals, empty, weeds, structs = {}, {}, [], [], []
        tasks = []
        d = self.day
        for pos, t in self._cells():
            if t == "LOCKED":
                continue
            if t is None:
                empty.append(pos)
                continue
            k = t.get("kind")
            if k == "WEED":
                weeds.append(pos)
            elif k == "PLANT":
                c = t["crop"]
                crops[c] = crops.get(c, 0) + 1
                seed, first, maxd, interval, maxy, ongoing = CROPS[c]
                age = d - t["planted_day"]
                if not t["watered_today"]:
                    # Water only when it pays: a plant that missed yesterday dies tonight (must); one-time crops gain
                    # yield only inside their bonus window; ongoing crops gain nothing from water unless fertilized.
                    # Otherwise every other day is enough (death needs two dry days in a row).
                    in_window = (not ongoing) and (maxd + 1) // 2 <= age <= maxd
                    fert = t.get("fertilized_until_day", -1) >= d
                    if t["consecutive_unwatered"] >= 1:
                        tasks.append((12, "WATER", pos))
                    elif in_window or (ongoing and fert):
                        tasks.append((8, "WATER", pos))
                    elif self.p["water_daily"]:
                        tasks.append((self.p["water_daily"], "WATER", pos))
                if t.get("yield_units", 0) > 0 and age >= first:
                    cap = maxy if t.get("fertilized_until_day", -1) >= 0 or c == "MELON" else min(maxy, 1 + (maxd - (maxd + 1) // 2 + 1))
                    if not ongoing and (age >= maxd or t.get("yield_units", 0) >= cap):
                        # harvest as soon as the yield is maxed (melon hits 6 at age 10, not 12): earlier cash,
                        # before the market fills up; decays from day planted+max_yield_day+1 on
                        tasks.append((12, "HARVEST", pos))
                    elif ongoing:
                        tasks.append((9 + t.get("yield_units", 0), "HARVEST", pos))   # held yield caps at 4
                if t.get("fertilized_until_day", -1) < d:
                    if not ongoing and (maxd + 1) // 2 - 1 <= age < maxd:
                        tasks.append((5, "FERTILIZE", pos))       # doubles the daily bonus in the yield window
                    elif ongoing and first - 1 <= age <= first + interval * maxy:
                        tasks.append((6, "FERTILIZE", pos))       # 2 units per production while fertilized
            elif "animal" in t:
                a = t["animal"]
                animals[a] = animals.get(a, 0) + 1
                if not t["fed_today"]:
                    tasks.append((10 + 4 * t["consecutive_unfed"], "FEED", pos))
                elif not t["cared_today"]:
                    tasks.append((9, "CARE", pos))
                if t.get("yield_units", 0) >= 2 or (t.get("yield_units", 0) >= 1 and self.hour >= 16):
                    tasks.append((7, "HARVEST", pos))
                if t.get("fertilizer_available"):
                    tasks.append((6, "COLLECT_FERTILIZER", pos))
            elif k in ("COOP", "PASTURE"):
                structs.append((pos, k))
        return crops, animals, empty, weeds, structs, tasks

    # ------------------------------------------------------------------ plan: what to plant / build today
    def _plan_tiles(self, crops, animals, empty, structs):
        phase_day, ctarget, atarget = self._phase()
        plan = {}
        free_struct = {"COOP": [p for p, k in structs if k == "COOP"],
                       "PASTURE": [p for p, k in structs if k == "PASTURE"]}
        # structures for animals we still want (build near the shed)
        want_struct = {"COOP": 0, "PASTURE": 0}
        for a, n in atarget.items():
            if self.day > self.p["last_animal_day"][a]:
                continue
            have = animals.get(a, 0) + self.shed.get(a, 0) + sum(i.get(a, 0) for i in self.invs)
            want_struct[ANIMALS[a][1]] += max(0, n - have)
        ordered = sorted(empty, key=self._shed_dist)
        for kind in ("PASTURE", "COOP"):
            need = max(0, want_struct[kind] - len(free_struct[kind]))
            for pos in ordered[:]:
                if need <= 0:
                    break
                plan[pos] = "BUILD_" + kind
                ordered.remove(pos)
                need -= 1
        # crops: most-below-target crop that can still mature, but never more live plants than the crew can water
        counts = dict(crops)
        live = sum(crops.values())
        crew = min(15, max(4, len(self.units)))
        capacity = crew * self.p["plants_per_unit"]   # every-other-day watering outside yield windows
        for pos in reversed(ordered):
            if live >= capacity:
                break
            live += 1
            best, gap = None, 0
            for c, n in ctarget.items():
                if not _age_ok(c, self.day):
                    continue
                g = n - counts.get(c, 0)
                if g > gap:
                    best, gap = c, g
            if best is None:
                if _age_ok("WHEAT", self.day):
                    best = "WHEAT"
                elif _age_ok("CARROT", self.day):
                    best = "CARROT"
                else:
                    continue
            plan[pos] = best
            counts[best] = counts.get(best, 0) + 1
            if best == "MELON":
                self._melons_today = getattr(self, "_melons_today", 0) + 1
        return plan, free_struct, want_struct

    # ------------------------------------------------------------------ unit actions
    def _unit_actions(self, tasks, plan, free_struct):
        acts = []
        taken = set()
        seeds = dict(self.seeds)
        wheat_shed = self.shed.get("WHEAT", 0)
        fert_shed = self.shed.get("FERTILIZER", 0)
        animals_in_shed = {a: self.shed.get(a, 0) for a in ANIMALS}
        for pos, what in plan.items():
            if what.startswith("BUILD_"):
                tasks.append((5, what, pos))
            elif seeds.get(what, 0) > 0:
                tasks.append((8 if self.hour < 20 else 2, "PLANT_" + what, pos))
        for pos, t in self._cells():
            if isinstance(t, dict) and t.get("kind") == "WEED":
                tasks.append((5, "DIG", pos))
        carried_animals = {a: sum(i.get(a, 0) for i in self.invs) for a in ANIMALS}
        for a in ANIMALS:
            n = animals_in_shed.get(a, 0) + carried_animals[a]
            for pos in free_struct[ANIMALS[a][1]][:n]:
                tasks.append((11, "PLACE_" + a, pos))
        late = self.hour >= TPD - 2
        # spatial zones: split the columns of the board among units so they do not criss-cross the farm
        nu = len(self.units)
        zone = {ui: (ui * self.n // nu, (ui + 1) * self.n // nu) for ui in range(nu)}
        if self.assign_day != self.day:
            self.assign, self.assign_day = {}, self.day
        valid = {(k, p) for _, k, p in tasks}
        for ui in list(self.assign):
            if ui >= len(self.units) or self.assign[ui] not in valid:
                del self.assign[ui]
        taken |= set(self.assign.values())
        for ui, pos in enumerate(self.units):
            inv = self.invs[ui]
            carrying = sum(v for k, v in inv.items() if k not in ANIMALS)
            at_shed = pos in self.shed_tiles
            # --- deliver produce / animals to the shed when loaded or at day end
            if at_shed and carrying and (carrying >= 6 or late or self.hour >= 18):
                room = SHED_CAP - self.shed_used
                if room > 0:
                    acts.append(["DROP"])
                    self.shed_used += carrying
                    continue
            best, best_score = None, -1e9
            committed = self.assign.get(ui)
            for pr, kind, tpos in tasks:
                if (kind, tpos) in taken and committed != (kind, tpos):
                    continue
                need_item = None
                if kind == "FEED":
                    need_item = "WHEAT"
                elif kind == "FERTILIZE":
                    need_item = "FERTILIZER"
                elif kind.startswith("PLACE_"):
                    need_item = kind[6:]
                if need_item and inv.get(need_item, 0) <= 0:
                    avail = {"WHEAT": wheat_shed, "FERTILIZER": fert_shed}.get(need_item, animals_in_shed.get(need_item, 0))
                    if avail <= 0:
                        continue
                    d = self._shed_dist(pos) + min(self._dist(s, tpos) for s in self.shed_tiles)
                else:
                    d = self._dist(pos, tpos)
                lo, hi = zone[ui]
                if nu > 2 and not (lo <= tpos[0] < hi):
                    d += self.p["zone_penalty"]
                score = pr * 2 - d + (6 if committed == (kind, tpos) else 0)   # stick to the committed task
                if tpos == pos and not need_item:
                    score += 20                                  # finish every job on the tile we stand on
                if score > best_score:
                    best, best_score = (pr, kind, tpos, need_item), score
            if best is None:
                if carrying and not late:
                    act = self._move_toward(pos, min(self.shed_tiles, key=lambda s: self._dist(pos, s)))
                    acts.append(act or ["DROP"])
                else:
                    acts.append(["PASS"])
                continue
            pr, kind, tpos, need_item = best
            taken.add((kind, tpos))
            self.assign[ui] = (kind, tpos)
            if need_item and inv.get(need_item, 0) <= 0:
                # fetch the item first
                if at_shed:
                    if need_item == "WHEAT":
                        lo, hi = zone[ui]
                        unfed = sum(1 for _, k, p in tasks if k == "FEED" and lo <= p[0] < hi)
                        n = max(1, min(wheat_shed, 8, max(unfed, 2)))
                        wheat_shed -= n
                    elif need_item == "FERTILIZER":
                        n = min(fert_shed, 3)
                        fert_shed -= n
                    else:
                        n = 1
                        animals_in_shed[need_item] -= 1
                    acts.append(["PICKUP", need_item, n])
                else:
                    acts.append(self._move_toward(pos, min(self.shed_tiles, key=lambda s: self._dist(pos, s))))
                continue
            if pos != tpos:
                acts.append(self._move_toward(pos, tpos))
                continue
            if kind.startswith("PLANT_"):
                c = kind[6:]
                if seeds.get(c, 0) > 0:
                    seeds[c] -= 1
                    acts.append(["PLANT", c])
                else:
                    acts.append(["PASS"])
            elif kind.startswith("PLACE_"):
                acts.append(["PLACE", kind[6:]])
            else:
                acts.append([kind])
        return acts

    # ------------------------------------------------------------------ market
    def _market(self, crops, animals, empty, plan, want_struct, free_struct):
        orders = []
        money = self.money - self.p["reserve"]
        d, h = self.day, self.hour
        # land on the top teams' schedule
        k = len(self.farm["unlocked_quadrants"]) - 1
        if k < 3 and d >= self.p["land_days"][k] and money >= LAND[k] + 1200:
            orders.append(["BUY_LAND"])
            money -= LAND[k]
        # hires: enough hands for today's work
        if h <= 1:
            work = 3 * sum(crops.values()) + 8 * sum(animals.values()) + 4 * len(plan)   # unit-steps per day
            target = int(min(self.p["hire_cap"], max(3, math.ceil(work / self.p["hire_div"]) - 1)))  # fib-priced hands
            have = len(self.units) - 1
            cost, n_today = 0, self.farm.get("hires_today", 0)
            fib = [1, 1]
            while len(fib) < 30:
                fib.append(fib[-1] + fib[-2])
            while have < target and self.money > fib[n_today] + 20:
                orders.append(["HIRE"])
                money -= fib[n_today]
                n_today += 1
                have += 1
        # seeds for today's plan (planted from next step on)
        need = {}
        for pos, what in plan.items():
            if what in CROPS:
                need[what] = need.get(what, 0) + 1
        for c, n in need.items():
            q = min(n - self.seeds.get(c, 0), int(money // CROPS[c][0]))
            if q > 0 and h < 21:
                orders.append(["BUY_SEED", c, q])
                money -= q * CROPS[c][0]
        # animals for free structures, only with money left after seeds for every planned crop tile
        seed_budget = sum(CROPS[w][0] for w in plan.values() if w in CROPS)
        money -= seed_budget
        for a, (cost, kind, _) in ANIMALS.items():
            if d > self.p["last_animal_day"][a]:
                continue
            free = len(free_struct[kind]) - self.shed.get(a, 0)
            if free <= 0:
                continue
            _, _, atarget = self._phase()
            have = animals.get(a, 0) + self.shed.get(a, 0)
            q = min(free, max(0, atarget.get(a, 0) - have), int(money // cost))
            if q > 0 and self.shed_used + q < SHED_CAP:
                orders.append(["BUY_ANIMAL", a, q])
                money -= q * cost
        # feed wheat: keep 3 days of feed; buy only the shortfall that our own ripe wheat will not cover today
        n_an = sum(animals.values())
        wheat_have = self.shed.get("WHEAT", 0) + sum(i.get("WHEAT", 0) for i in self.invs)
        ripe = sum(t.get("yield_units", 0) for _, t in self._cells() if isinstance(t, dict)
                   and t.get("crop") == "WHEAT" and self.day - t["planted_day"] >= CROPS["WHEAT"][2])
        need = n_an + 1 - wheat_have - ripe                      # today's feeding
        if need > 0 and self.shed_used < SHED_CAP - 5:
            q = min(need, int(max(0, self.money - 50) // max(1, self.prices.get("WHEAT", 25) + 2)))
            if q > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", q])
        # selling: demand-aware lots at the preferred hours; more when the shed fills; liquidate at the end
        sells = self._sells(n_an)
        return (sells + orders)[:10]

    def _sells(self, n_an):
        """Demand-preserving selling relative to the CURRENT quote: each step sell at most what keeps our own
        price impact within `impact` of the quote (the rival may already have pushed the price below base);
        preferred hours unless the shed is filling; fertilizer only at a decent price; full liquidation at
        the end of the season."""
        h = self.hour
        pressure = self.shed_used / SHED_CAP
        final = self.step >= (DAYS - 1) * TPD + 8
        if not final and h not in self.p["sell_hours"] and pressure < 0.4:
            return []
        impact = 0.15 if pressure < 0.6 else 0.3 if pressure < 0.85 else 0.6
        out = []
        inv = dict(self.minv)
        for item, (base, *_rest) in MARKET_PARAMS.items():
            stock = self.shed.get(item, 0)
            if item == "WHEAT" and not final:
                stock -= 3 * n_an                                  # never sell the feed we would buy back
            if stock <= 0:
                continue
            p0 = self.prices.get(item, base)
            if final:
                floor = 0 if self.step >= DAYS * TPD - 4 else 0.5 * p0
            elif item == "FERTILIZER":
                floor = max(self.p["fert_min_price"] * (1 - pressure), (1 - impact) * p0)
            else:
                floor = (1 - impact) * p0
            q = sellable(item, inv.get(item, I0), stock, floor) if floor > 0 else stock
            if q > 0:
                out.append(["SELL", item, int(q)])
                inv[item] = inv.get(item, I0) + int(q)
        return out

    # ------------------------------------------------------------------ entry
    def __call__(self, obs, config=None):
        try:
            self._parse(obs)
            crops, animals, empty, weeds, structs, tasks = self._census()
            plan, free_struct, want_struct = self._plan_tiles(crops, animals, empty, structs)
            units = self._unit_actions(tasks, plan, free_struct)
            market = self._market(crops, animals, empty, plan, want_struct, free_struct)
            units = [u if isinstance(u, list) and u else ["PASS"] for u in units]
            return {"farmer": units[0], "hands": units[1:], "market": market}
        except Exception:
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in obs["farms"][int(obs["player"])]["hands"]],
                    "market": []}


_AGENT = None


def agent(observation, configuration=None):
    global _AGENT
    if _AGENT is None or int(observation["step"]) == 0:
        _AGENT = TopPlanner()
    return _AGENT(observation, configuration)
