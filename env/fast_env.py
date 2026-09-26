"""Fast local Kaggriculture environment.

"Code as environment" (CodeMidas): we drive the *official* interpreter from
kaggle_environments directly, skipping the framework's per-step schema
validation / structify copies. Game logic is therefore bit-identical to the
online judge; `env/replay_check.py` verifies this against downloaded replays.
"""
import copy
import importlib
import pickle

from kaggle_environments.utils import Struct

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")

DEFAULT_CONFIG = {
    "episodeSteps": 720, "actTimeout": 1, "runTimeout": 1200, "boardSize": 10,
    "startingMoney": 3000, "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24,
    "shedCapacity": 100, "weedSpawnChance": 0.005, "townShopUnlockInterval": 3,
    "townShopSellInterval": 4, "townCenterSellInterval": 24, "farmHandCostMult": 1,
    "marketParams": {},
}
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _sanitize(action):
    """Mirror the framework: anything that is not a dict becomes the default action."""
    return action if isinstance(action, dict) else PASS_ACTION


class FarmEnv:
    def __init__(self, seed, config=None):
        self.seed = int(seed)
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.reset()

    def reset(self):
        cfg = Struct(**self.config)
        cfg.seed = self.seed
        self.env = Struct(configuration=cfg, info={}, done=False)
        self.state = [
            Struct(observation=Struct(step=0, remainingOverageTime=60, player=i),
                   action=None, reward=0, status="ACTIVE", info={})
            for i in range(2)
        ]
        K.interpreter(self.state, self.env)  # _initialize
        self.step_count = 0
        self.done = False
        return self

    # -- observations -----------------------------------------------------
    def obs(self, i, copy_obs=True):
        o0 = self.state[0].observation
        oi = self.state[i].observation
        o = {
            "player": i, "step": self.step_count, "remainingOverageTime": 60,
            "day": o0.day, "hour": o0.hour, "farms": o0.farms, "market": o0.market,
            "town": o0.town, "private": oi.private,
        }
        # pickle round trip = deepcopy for this plain data (dict/list/int/str), ~5x faster
        return pickle.loads(pickle.dumps(o, pickle.HIGHEST_PROTOCOL)) if copy_obs else o

    @property
    def money(self):
        return [f["money"] for f in self.state[0].observation.farms]

    # -- dynamics -----------------------------------------------------------
    def step(self, a0, a1):
        assert not self.done
        for s, a in zip(self.state, (a0, a1)):
            s.action = _sanitize(a)
        self.state[0].observation.step = self.step_count
        K.interpreter(self.state, self.env)
        self.step_count += 1
        if self.state[0].status == "DONE" or self.step_count >= self.config["episodeSteps"] - 1:
            self.done = True
        return self.done

    def rewards(self):
        return [s.reward for s in self.state]

    # -- cloning (for prefix rollouts / search) ------------------------------
    def clone(self):
        new = FarmEnv.__new__(FarmEnv)
        new.seed, new.config = self.seed, self.config
        new.env = copy.deepcopy(self.env)
        new.state = copy.deepcopy(self.state)
        new.step_count, new.done = self.step_count, self.done
        return new


def play(agent0, agent1, seed, copy_obs=True, max_steps=None):
    """Play one full game between two callables obs -> action. Returns (money0, money1)."""
    env = FarmEnv(seed)
    n = max_steps or env.config["episodeSteps"]
    while not env.done and env.step_count < n:
        a0 = agent0(env.obs(0, copy_obs))
        a1 = agent1(env.obs(1, copy_obs))
        env.step(a0, a1)
    return env.money
