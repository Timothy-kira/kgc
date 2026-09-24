"""Execution-consistency check (CodeMidas §3.3 analogue).

Re-simulates an online replay from (seed, both players' actions) with the local
FarmEnv and asserts that every recorded step matches: money, tiles, market,
town, sheds. Usage:
    python -m env.replay_check <episode_id | replay.json> [...]
"""
import json
import sys
import urllib.request

from env.fast_env import FarmEnv

REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{}.json"


def load_replay(src):
    if str(src).endswith(".json"):
        with open(src) as f:
            return json.load(f)
    with urllib.request.urlopen(REPLAY_URL.format(src), timeout=300) as r:
        return json.load(r)


def replay_actions(rep):
    """actions[t] = (a0, a1) applied at transition t -> t+1."""
    steps = rep["steps"]
    return [(steps[t + 1][0].get("action"), steps[t + 1][1].get("action")) for t in range(len(steps) - 1)]


def _norm(x):
    return json.loads(json.dumps(x))


def check(rep, verbose=True):
    seed = rep["info"]["seed"]
    cfg = {k: v for k, v in rep["configuration"].items() if k in (
        "episodeSteps", "boardSize", "startingMoney", "maxMarketOrdersPerTurn", "turnsPerDay",
        "shedCapacity", "weedSpawnChance", "townShopUnlockInterval", "townShopSellInterval",
        "townCenterSellInterval", "farmHandCostMult", "marketParams")}
    env = FarmEnv(seed, cfg)
    steps = rep["steps"]
    for t, (a0, a1) in enumerate(replay_actions(rep)):
        env.step(a0, a1)
        rec = steps[t + 1]
        ro = rec[0]["observation"]
        mine = env.obs(0, copy_obs=False)
        for key in ("farms", "market", "town"):
            if _norm(mine[key]) != _norm(ro[key]):
                if verbose:
                    print(f"step {t + 1}: mismatch in {key}")
                return False, t + 1
        for i in range(2):
            if _norm(env.state[i].observation.private) != _norm(rec[i]["observation"]["private"]):
                if verbose:
                    print(f"step {t + 1}: mismatch in private[{i}]")
                return False, t + 1
    final = [s.get("reward") for s in steps[-1]]
    ok = [round(m) for m in env.money] == [round(r or 0) for r in final]
    if verbose:
        print(f"episode {rep['info'].get('EpisodeId')}: steps={len(steps)} local={env.money} online={final} ok={ok}")
    return ok, len(steps)


if __name__ == "__main__":
    allok = True
    for src in sys.argv[1:]:
        ok, _ = check(load_replay(src))
        allok &= ok
    sys.exit(0 if allok else 1)
