"""Parametric rule-based market adversaries (dynamic opponents with distinct selling styles).

Each wraps the base executor and rewrites its SELL orders with a simple style, with
randomised parameters per game so the league sees a spread of market behaviour.
"""
import numpy as np

from agent.controller import MarketController
from agent.features import BASE_BIN


def make_rule_adversary(name, base, rng):
    thr = float(rng.uniform(0.6, 1.1))
    late = float(rng.uniform(0.03, 0.2))

    def front_run(P, G, tr, base_bins):
        # sell premium goods as soon as any stock exists (race the opponent to the high prices)
        b = np.full(9, BASE_BIN)
        for k in range(9):
            if P[k, 22] > 0 and P[k, 2] > 0 and P[k, 0] >= thr * 0.8:
                b[k] = 4
        return b, None

    def hoarder(P, G, tr, base_bins):
        # hold premium goods while price is below thr*base, dump late
        b = np.full(9, BASE_BIN)
        for k in range(9):
            if P[k, 22] > 0 and P[k, 0] < thr and G[2] > late:
                b[k] = 0
            elif G[2] <= late and P[k, 2] > 0:
                b[k] = 4
        return b, None

    def dripper(P, G, tr, base_bins):
        b = np.full(9, BASE_BIN)
        for k in range(9):
            if base_bins[k] == 4 and G[2] > late:
                b[k] = 3 if rng.random() < 0.5 else 2
        return b, None

    def noisy(P, G, tr, base_bins):
        b = np.full(9, BASE_BIN)
        for k in range(9):
            if P[k, 2] > 0 and rng.random() < 0.03:
                b[k] = int(rng.integers(0, 5))
        return b, None

    styles = {"front_run": front_run, "hoarder": hoarder, "dripper": dripper, "noisy": noisy}
    return MarketController(base, styles[name])
