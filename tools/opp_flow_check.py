"""Can a live agent see the opponent's market flows (the TTT labels) during a real match?

python -m tools.opp_flow_check [seeds=9301,9302,9303]
Infers the opponent's inventory-moving market flow per step and product from exactly what a Kaggle agent
observes -- public market inventory, public town shops (-> deterministic drain), its own orders and its own
private farm/shed (own unit phase replayed with the official rule code; $1 sales add no supply) -- and compares
with the engine's ground truth (instrumented _commit_unit). cha22 (seat 0, 'us') vs cha22 (seat 1).
Result 2026-09-25: 99.87% of (step, product) exact; error <= 1.2% of volume except WOOL (11%, $1-floor
interleaving inside a step, where the price does not move anyway).
"""
import collections
import copy
import importlib
import sys
from agent.loader import load_agent
from env.fast_env import FarmEnv

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
P = K.PRODUCTS
TRUE = collections.defaultdict(lambda: collections.Counter())   # step -> Counter[(player, item)] inventory delta
_orig = K._commit_unit
CUR = {"env": None, "t": 0}

def _patched(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig(op, item, price, farm, private, market, shed_capacity)
    if ok and op in ("SELL", "BUY_PRODUCT"):
        pl = 0 if farm is CUR["env"].state[0].observation.farms[0] else 1
        d = (1 if price > 1 else 0) if op == "SELL" else -1
        TRUE[CUR["t"]][(pl, item)] += d
    return ok
K._commit_unit = _patched

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

def run(seed):
    a, b = load_agent("league/cha22.py"), load_agent("league/cha22.py")
    env = FarmEnv(seed); CUR["env"] = env; TRUE.clear()
    exact = total = 0; abs_err = collections.Counter(); vol = collections.Counter()
    while not env.done:
        o0, o1 = env.obs(0), env.obs(1)
        act0 = a(o0, env.config); act1 = b(o1, env.config)
        t = env.step_count; CUR["t"] = t
        inv_before = dict(o0["market"]["inventory"]); shops = list(o0["town"]["unlocked_shops"])
        shed0 = dict(o0["private"]["shed"])
        env.step(act0, act1)
        inv_after = env.obs(0, copy_obs=False)["market"]["inventory"]
        # our own flow as a live agent knows it: replay OUR unit actions on a copy of our own (public farm +
        # private shed) with the official rule code -> shed after the unit phase; SELL executes up to that.
        farm_c = copy.deepcopy(o0["farms"][0]); priv_c = copy.deepcopy(o0["private"])
        day, tpd = o0["day"], 24
        units = [act0.get("farmer", ["PASS"])] + list(act0.get("hands") or [])
        demand = collections.Counter(u[1] for u in units if isinstance(u, list) and len(u) >= 2 and u[0] == "PLANT")
        seeds = priv_c.get("seeds", {})
        blocked = {c for c, n in demand.items() if n > seeds.get(c, 0)}
        for i, u in enumerate(units):
            u = ["PASS"] if (isinstance(u, list) and len(u) >= 2 and u[0] == "PLANT" and u[1] in blocked) else u
            K._apply_unit_action(farm_c, priv_c, i, u, 10, day, tpd, 100)
        ours = collections.Counter(); left = dict(priv_c["shed"]); inv_sim = dict(inv_before)
        money = farm_c["money"]; room = 100 - sum(priv_c["shed"].values())
        for m in act0.get("market") or []:
            if not (isinstance(m, list) and len(m) >= 3 and m[1] in P):
                continue
            it, q = m[1], int(m[2])
            if m[0] == "SELL":
                for _ in range(min(q, left.get(it, 0))):
                    left[it] -= 1
                    pr = K.market_price(it, inv_sim[it])
                    money += pr; room += 1
                    if pr > 1:                      # a $1 sale does not add supply (engine rule)
                        inv_sim[it] += 1; ours[it] += 1
            elif m[0] == "BUY_PRODUCT":
                for _ in range(q):
                    pr = K.market_price(it, inv_sim[it] - 1)
                    if money < pr or room <= 0:
                        break
                    money -= pr; room -= 1; inv_sim[it] -= 1; ours[it] -= 1
        dr = drain(t, shops)
        for it in P:
            inferred = (inv_after[it] - inv_before[it]) + dr[it] - ours[it]
            truth = TRUE[t][(1, it)]
            total += 1; exact += inferred == truth
            abs_err[it] += abs(inferred - truth); vol[it] += abs(truth)
    return exact, total, abs_err, vol, env.money

if __name__ == "__main__":
    E = T = 0; AE = collections.Counter(); V = collections.Counter()
    for seed in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "9301,9302,9303").split(",")]:
        e, t, ae, v, money = run(seed); E += e; T += t; AE.update(ae); V.update(v)
        print("seed", seed, "exact per (step, product):", f"{100*e/t:.2f}%", "money", [round(x) for x in money])
    print("overall exact:", f"{100*E/T:.2f}%")
    for it in P:
        print(f"{it:11s} opponent volume {V[it]:6d}   inferred abs error {AE[it]:5d}  ({100*AE[it]/max(1,V[it]):.1f}% of volume)")
