"""Open-loop replay opponents: a top player's recorded actions on its own seed.

On the episode's seed, the replayed player's farm evolves exactly as online
(its farm is private to it); only the shared market (and therefore money and
money-gated purchases) can differ because our agent sits in the other seat.
This gives a cheap, faithful evaluation / training opponent for "how would we
have done against DSM on that exact game".
"""
import json

from env.fast_env import FarmEnv


class ReplayOpponent:
    def __init__(self, actions, seat):
        self.actions = actions
        self.seat = seat

    def __call__(self, obs, config=None):
        t = int(obs["step"])
        if t < len(self.actions):
            a = self.actions[t][self.seat]
            return a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}


def play_vs_replay(agent, row, replay_seat, copy_obs=True):
    """Our `agent` takes seat 1-replay_seat. Returns (our money, replay money, online money of replay seat,
    online money of the seat we replaced)."""
    acts = row["actions"]
    cfg = {k: v for k, v in json.loads(row["config"]).items() if v is not None}
    env = FarmEnv(row["seed"], cfg)
    opp = ReplayOpponent(acts, replay_seat)
    me = 1 - replay_seat
    while not env.done:
        o_me = env.obs(me, copy_obs)
        o_op = env.obs(replay_seat, copy_obs)
        a_me = agent(o_me)
        a_op = opp(o_op)
        env.step(*((a_me, a_op) if me == 0 else (a_op, a_me)))
    m = env.money
    return m[me], m[replay_seat], row[f"reward_{replay_seat}"], row[f"reward_{me}"]
