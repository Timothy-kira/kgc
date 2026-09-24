"""Probe: how much do sell decisions alone move the outcome?

Plays base+controller(variant) vs plain base on several seeds (both seats) and
reports money difference. Usage: python -m tools.sell_probe [n_seeds]
"""
import multiprocessing as mp
import sys

import numpy as np

from agent.controller import MarketController
from env.fast_env import play
from tools.arena import load

BASE = "/home/user/kgc/league/metav4.py"


def pol_all(P, G, tr, base_bins):
    return np.where(P[:, 2] > 0, 4, 0), None


def pol_hold_premium(P, G, tr, base_bins):
    b = np.full(9, 5)
    late = G[2] < 0.1
    for k in range(9):
        if P[k, 22] > 0 and P[k, 0] < 0.8 and not late:   # premium product below 80% of base price: hold
            b[k] = 0
    return b, None


def pol_drip(P, G, tr, base_bins):
    b = np.full(9, 5)
    for k in range(9):
        if base_bins[k] == 4 and P[k, 22] > 0 and G[2] > 0.1:
            b[k] = 3
    return b, None


def pol_base(P, G, tr, base_bins):
    return np.full(9, 5), None


VARIANTS = {"base_via_ctrl": pol_base, "sell_all": pol_all, "hold_premium": pol_hold_premium, "drip": pol_drip}


def job(args):
    name, seed, seat = args
    me = MarketController(load(BASE), VARIANTS[name])
    opp = load(BASE)
    if seat == 0:
        m = play(me, opp, seed)
        return name, seed, seat, m[0] - m[1]
    m = play(opp, me, seed)
    return name, seed, seat, m[1] - m[0]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(v, 2000 + s, seat) for v in VARIANTS for s in range(n) for seat in (0, 1)]
    with mp.Pool(4) as p:
        res = p.map(job, jobs)
    for v in VARIANTS:
        d = [r[3] for r in res if r[0] == v]
        print(f"{v:14s} mean diff {np.mean(d):9.0f}  wins {sum(x > 0 for x in d)}/{len(d)}  {np.round(d).tolist()}")
