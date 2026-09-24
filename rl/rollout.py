"""Rollout workers: play one game with the (sampling) controller policy and record a trace."""
import numpy as np

from agent.controller import MarketController
from agent.features import N_TGT
from env.fast_env import FarmEnv
from model.policy_np import NumpyPolicy

BASE_EXECUTOR = "/home/user/kgc/league/metav4.py"
_MODULE_CACHE = {}


def load_agent(path):
    from tools.arena import load
    return load(path)


def make_opponent(spec, rng):
    """spec: path/to/agent.py | ctrl:<weights.npz>[@temp] | rule:<name>"""
    if spec.startswith("ctrl:"):
        wpath, _, temp = spec[5:].partition("@")
        pol = NumpyPolicy(wpath, temperature=float(temp or 1.0), rng=rng)
        return MarketController(load_agent(BASE_EXECUTOR), pol.act_sample if temp else pol.act_greedy)
    if spec.startswith("rule:"):
        from rl.rule_adversaries import make_rule_adversary
        return make_rule_adversary(spec[5:], load_agent(BASE_EXECUTOR), rng)
    return load_agent(spec)


def rollout(args):
    """args: dict(weights, temperature, opponent, seed, seat, greedy, rng_seed, base_only)"""
    rng = np.random.default_rng(args.get("rng_seed"))
    seat = args["seat"]
    if args.get("base_only"):
        me = MarketController(load_agent(BASE_EXECUTOR), None, record=False)
        pol = None
    else:
        pol = NumpyPolicy(args["weights"], temperature=args.get("temperature", 1.0), rng=rng)
        me = MarketController(load_agent(BASE_EXECUTOR), pol.act_greedy if args.get("greedy") else pol.act_sample,
                              record=True)
    opp = make_opponent(args["opponent"], rng)
    env = FarmEnv(args["seed"])
    while not env.done:
        o0 = env.obs(0)
        o1 = env.obs(1)
        if seat == 0:
            env.step(me(o0), opp(o1))
        else:
            env.step(opp(o0), me(o1))
    money = env.money
    out = {"seed": args["seed"], "seat": seat, "opponent": args["opponent"],
           "money_me": money[seat], "money_opp": money[1 - seat], "tag": args.get("tag")}
    if pol is None or not me.trace:
        return out
    T = len(me.trace)
    P = np.stack([t[0] for t in me.trace]).astype(np.float32)
    G = np.stack([t[1] for t in me.trace]).astype(np.float32)
    bins = np.stack([t[4] for t in me.trace]).astype(np.int64)
    logp = np.stack([t[5][0] for t in me.trace]).astype(np.float32)
    mask = np.stack([t[5][2] for t in me.trace])
    base_bins = np.stack([t[3] for t in me.trace]).astype(np.int64)
    n_days = (T + 23) // 24
    day_tgt = np.zeros((n_days, N_TGT), np.float32)
    day_mask = np.zeros(n_days, np.float32)
    for t in me.trace:
        for d, tgt in t[5][3]:
            if d < n_days:
                day_tgt[d] = tgt
                day_mask[d] = 1.0
    out.update(P=P, G=G, bins=bins, logp=logp, mask=mask, base_bins=base_bins, day_tgt=day_tgt,
               day_mask=day_mask, T=T)
    return out
