"""Upper bound of opponent prediction: cha22 with its front_run hook fed the opponent's TRUE future actions."""
import json, sys
from agent.loader import load_module, entry_name, call_adapter, load_agent
from env.fast_env import FarmEnv
from infra.fork import warm_pool, best_effort

def cha22(front_run=False, plan=None):
    m = load_module("league/cha22.py")
    if front_run:
        m._IMPL.chassis.cfg["front_run"] = True
        m._IMPL.chassis.opponent_plan = plan
    return call_adapter(getattr(m, entry_name(m)))

def path(n): return "league/cha22.py" if n == "cha22" else f"league/pub/{n}/main.py"

def play(job):
    opp, seed = job
    best_effort()
    # 1) baseline game, record the opponent's actions
    me, op = cha22(), load_agent(path(opp)); env = FarmEnv(seed); rec = []
    while not env.done:
        a = me(env.obs(0), env.config); b = op(env.obs(1), env.config); rec.append(b); env.step(a, b)
    base = env.money[0] - env.money[1]
    # 2) same game, our cha22 front-runs with the opponent's recorded (true) plan
    me, op = cha22(True, rec), load_agent(path(opp)); env = FarmEnv(seed)
    while not env.done:
        a = me(env.obs(0), env.config); b = op(env.obs(1), env.config); env.step(a, b)
    return opp, seed, base, env.money[0] - env.money[1], env.money[0]

if __name__ == "__main__":
    jobs = [(o, 9800 + s) for o in ("cha22", "prvsiyan_frontier", "hakfield", "tetsutani_ms") for s in range(3)]
    with warm_pool(4, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(play, jobs):
            print(json.dumps({"opp": r[0], "seed": r[1], "diff_base": round(r[2]), "diff_frontrun_oracle": round(r[3]), "gain": round(r[3] - r[2])}), flush=True)
