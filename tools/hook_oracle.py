"""G1: headroom of cha22's insertion points (docs/PLAN_v4.md §2, §8).

python -m tools.hook_oracle h1 [seeds=10] [procs=4] [first_seed=9401]
python -m tools.hook_oracle h5 [seeds=6] [procs=4] [first_seed=9401]

H1 (sell timing / quantity). Our side is cha22 in every variant; only its SELL orders for the controlled products
are replaced by a perfect-foresight argmin seller. Market decomposability (PLAN_v4 §1.4): our m-th unit sold at t
earns p(E_t + m) with E = opponent's inventory-moving flow minus town drain (our own sales excluded), so a unit is
sold at the step where E reaches its minimum over the rest of the game. Constraints: the 100-unit shed (hour-23
overflow guard, daytime holding cap), the $1 floor, everything sold by the last acting step, and cash: cha22's
tape funds its purchases from its own sales (days 0-10 run at 8..1,290 money), so below `min_money` the step
keeps cha22's sales (without this guard holding broke the animal purchases: 8 animals instead of 16). Wheat and fertilizer
are cha22's production inputs (FEED / FERTILIZE pick them up from the shed): 'oracle7' leaves both to cha22,
'oracle8' also times fertilizer above a reserve. Opponents: the baseline game's opponent replayed open-loop
('ol': exact foresight, the market is the only channel) and a live cha22 ('cl': it reacts, e.g. RACE), with the
baseline's opponent flows used as the forecast.

H5 (route choice). Fork at step 144 (all routes share steps 0-143 and the terminal from 648) over cha22's routes,
live cha22 opponent; best route vs the route cha22's shop table picks.
"""
import collections
import copy
import importlib
import json
import sys
import time

from agent.loader import call_adapter, entry_name, load_agent, load_module
from env.fast_env import FarmEnv
from infra.fork import best_effort, fork_branches, warm_pool

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
P = list(K.PRODUCTS)
LAST = 718                                        # last acting step (the interpreter ends after step 718)
NON_INPUT = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
CHA22 = "league/cha22.py"

# ----------------------------------------------------------------------------- engine flow tap
_orig_commit = K._commit_unit
TAP = {"env": None, "rec": None}


def _tap_commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig_commit(op, item, price, farm, private, market, shed_capacity)
    rec, env = TAP["rec"], TAP["env"]
    if ok and rec is not None and op in ("SELL", "BUY_PRODUCT"):
        pl = 0 if farm is env.state[0].observation.farms[0] else 1
        rec[env.step_count][pl][item] += (1 if price > 1 else 0) if op == "SELL" else -1
    return ok


K._commit_unit = _tap_commit


def drain(step, shops):
    d = collections.Counter()
    if step % 4 == 0:
        for s in shops:
            prods = K.SHOPS[s]
            for it in prods:
                d[it] += 2 if len(prods) == 1 else 1
    if step % 24 == 0:
        for it in K.TOWN_CENTER_PRODUCTS:
            d[it] += 1
    return d


class Tape:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, obs, cfg=None):
        t = obs["step"]
        return copy.deepcopy(self.actions[t]) if t < len(self.actions) else {"farmer": ["PASS"], "hands": [], "market": []}


def record(seed, seat):
    """Baseline cha22 (seat) vs cha22: opponent actions, per-step flows of both players, shops, money."""
    me, opp = load_agent(CHA22), load_agent(CHA22)
    env = FarmEnv(seed)
    flows = [[collections.Counter(), collections.Counter()] for _ in range(721)]
    TAP["env"], TAP["rec"] = env, flows
    acts, shops = [], []
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        shops.append(list(o[0]["town"]["unlocked_shops"]))
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        acts.append(copy.deepcopy(b))
        env.step(*((a, b) if seat == 0 else (b, a)))
    TAP["rec"] = None
    X = {p: [0] * (len(shops) + 1) for p in P}     # cumulative exogenous change before step t
    for t, sh in enumerate(shops):
        d = drain(t, sh)
        for p in P:
            X[p][t + 1] = X[p][t] + flows[t][1 - seat][p] - d[p]
    return {"opp_actions": acts, "X": X, "money": env.money[seat], "opp_money": env.money[1 - seat]}


class OracleSeller:
    """Replace cha22's SELLs of `products` by the argmin rule on a known exogenous path X."""

    def __init__(self, X, products, reserve=None, max_hold=60, margin=10, min_money=3000):
        self.X, self.products = X, list(products)
        self.min_money = min_money                  # cash guard: cha22's tape funds its purchases from its sales
        self.reserve = reserve or {}
        self.max_hold, self.margin = max_hold, margin
        self.sufmin = {}
        for p in self.products:
            s, n = [0] * (LAST + 2), len(X[p])
            s[LAST + 1] = float("inf")
            for t in range(LAST, -1, -1):
                s[t] = min(s[t + 1], X[p][t] if t < n else X[p][-1])
            self.sufmin[p] = s                                        # min over tau >= t of X[tau]

    def __call__(self, obs, action):
        t = int(obs["step"])
        if t < LAST and obs["farms"][int(obs["player"])]["money"] < self.min_money:
            return action                            # early game: every coin is reinvested, keep cha22's sales
        inv, shed = obs["market"]["inventory"], obs["private"]["shed"]
        carried = sum(sum(i.values()) for i in obs["private"].get("inventories") or [])
        want, gain = {}, {}
        for p in self.products:
            stock = shed.get(p, 0) - (0 if t >= LAST else self.reserve.get(p, 0))
            if stock <= 0:
                continue
            rel = self.sufmin[p][t + 1] - self.X[p][t] if t < LAST else 0      # best future exogenous change
            now = K.market_price(p, inv[p])
            gain[p] = K.market_price(p, inv[p] + min(0, rel)) - now          # price gain from waiting
            if t >= LAST or (rel >= 0 and now > 1):
                want[p] = stock
        left = {p: shed.get(p, 0) - self.reserve.get(p, 0) - want.get(p, 0) for p in gain}
        total = sum(shed.values()) - sum(want.values())

        def shed_units(n):
            for p in sorted(gain, key=lambda k: gain[k]):
                if n <= 0:
                    break
                k = min(n, max(0, left[p]))
                if k:
                    want[p] = want.get(p, 0) + k
                    left[p] -= k
                    n -= k
            return n

        if total > self.max_hold:                                          # daytime holding cap
            total -= (total - self.max_hold) - shed_units(total - self.max_hold)
        if t % 24 == 23 and total + carried > 100 - self.margin:          # end-of-day overflow guard
            shed_units(total + carried - (100 - self.margin))
        out, done = [], set()
        for o in action.get("market") or []:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in self.products:
                if o[1] in want and o[1] not in done:
                    out.append(["SELL", o[1], int(want[o[1]])])
                    done.add(o[1])
                continue
            out.append(o)
        for p, q in want.items():
            if p not in done and q > 0:
                out.append(["SELL", p, int(q)])
        action = dict(action)
        action["market"] = out[:10]
        return action


VARIANTS = {"oracle7": (NON_INPUT, {}), "oracle8": (NON_INPUT + ["FERTILIZER"], {"FERTILIZER": 10})}


def h1_job(job):
    seed, seat = job
    best_effort()
    rec = record(seed, seat)
    out = {"seed": seed, "seat": seat, "base_cl": rec["money"] - rec["opp_money"], "base_money": rec["money"]}
    for mode in ("ol", "cl"):
        for name, (prods, reserve) in [("base", (None, None))] + list(VARIANTS.items()):
            if mode == "cl" and name == "base":
                continue
            me = load_agent(CHA22)
            opp = Tape(rec["opp_actions"]) if mode == "ol" else load_agent(CHA22)
            seller = OracleSeller(rec["X"], prods, reserve) if prods else None
            env = FarmEnv(seed)
            while not env.done:
                o = [env.obs(0), env.obs(1)]
                a = me(o[seat], env.config)
                if seller:
                    a = seller(o[seat], a)
                b = opp(o[1 - seat], env.config)
                env.step(*((a, b) if seat == 0 else (b, a)))
            out[f"{name}_{mode}"] = env.money[seat] - env.money[1 - seat]
            out[f"{name}_{mode}_money"] = env.money[seat]
    return out


def h1(n_seeds, procs, first):
    jobs = [(first + s, seat) for s in range(n_seeds) for seat in (0, 1)]
    rows = []
    t0 = time.time()
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(h1_job, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    print(f"--- H1 over {len(rows)} games ({time.time() - t0:.0f}s): money / diff vs opponent / wins")
    for key in ["base_ol", "oracle7_ol", "oracle8_ol", "base_cl", "oracle7_cl", "oracle8_cl"]:
        d = [r[key] for r in rows if key in r]
        mk = key + "_money" if key != "base_cl" else "base_money"
        m = [r[mk] for r in rows if mk in r]
        print(f"{key:11s} money {sum(m) / len(m):9.0f}   diff {sum(d) / len(d):+8.0f}   wins {sum(x > 0 for x in d)}/{len(d)}")


def h5_job(job):
    seed, seat = job
    best_effort()
    mods = {}
    agents = []
    for side in (0, 1):
        m = load_module(CHA22)
        mods[side] = m
        agents.append(call_adapter(getattr(m, entry_name(m))))
    me, opp = agents[seat], agents[1 - seat]
    env = FarmEnv(seed)
    while env.step_count < 144:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    ch = mods[seat]._IMPL.chassis
    routes = sorted(ch.routes)
    base_router = ch.router
    shops = tuple(env.obs(0)["town"]["unlocked_shops"][:2])

    def branch(r):
        if r is not None:
            ch.router = lambda obs, step, st, r=r: (base_router(obs, step, st), r)[1] if 144 <= step < 648 else base_router(obs, step, st)
        chosen = None
        while not env.done:
            o = [env.obs(0), env.obs(1)]
            a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
            if chosen is None:
                chosen = ch.players.get(seat, {}).get("route")
            env.step(*((a, b) if seat == 0 else (b, a)))
        return chosen, env.money[seat] - env.money[1 - seat], env.money[seat]

    res = fork_branches([None] + routes, branch, max_parallel=1)
    default_route, d0, m0 = res[0]
    per = {r: v for r, v in zip(routes, res[1:]) if not isinstance(v, Exception)}
    best = max(per, key=lambda r: per[r][2])
    return {"seed": seed, "seat": seat, "shops": shops, "default_route": default_route, "default_money": m0,
            "default_diff": d0, "best_route": best, "best_money": per[best][2], "best_diff": per[best][1],
            "n_better": sum(v[2] > m0 for v in per.values()), "n_routes": len(per)}


def h5(n_seeds, procs, first):
    jobs = [(first + s, seat) for s in range(n_seeds) for seat in (0, 1)]
    rows = []
    t0 = time.time()
    with warm_pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(h5_job, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    n = len(rows)
    print(f"--- H5 over {n} games ({time.time() - t0:.0f}s)")
    print(f"default route: money {sum(r['default_money'] for r in rows) / n:.0f} diff {sum(r['default_diff'] for r in rows) / n:+.0f}"
          f" wins {sum(r['default_diff'] > 0 for r in rows)}/{n}")
    print(f"oracle route : money {sum(r['best_money'] for r in rows) / n:.0f} diff {sum(r['best_diff'] for r in rows) / n:+.0f}"
          f" wins {sum(r['best_diff'] > 0 for r in rows)}/{n}")


if __name__ == "__main__":
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (10 if mode == "h1" else 6)
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 9401
    (h1 if mode == "h1" else h5)(n, procs, first)
