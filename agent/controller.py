"""Market controller wrapped around a base executor agent.

The base agent (a strong scripted executor) plays the farm: planting, animals,
labour, purchases. The controller re-decides every SELL order: per product and
per step it picks a sell bin (see features.SELL_BINS). `policy(P, G, tracker)`
returns the bins; the default policy reproduces the base agent.
"""
import numpy as np

from agent.features import BASE_BIN, N_BINS, Tracker, bins_to_orders, label_bins


class MarketController:
    def __init__(self, base_agent, policy=None, record=False):
        self.base = base_agent
        self.policy = policy
        self.tracker = Tracker()
        self.record = record
        self.trace = []   # (P, G, stock, base_bins, chosen_bins, extra)

    def __call__(self, obs, config=None):
        act = self.base(obs, config) if config is not None else self.base(obs)
        if not isinstance(act, dict):
            return act
        try:
            return self._control(obs, act)
        except Exception:
            return act  # never lose a game to a controller bug

    def _control(self, obs, act):
        self.tracker.update(obs)
        base_market = act.get("market") or []
        P, G, stock, base_q = self.tracker.features(base_market)
        base_bins = label_bins(base_market, stock)          # what the base did, in bin terms (feature)
        if self.policy is None:
            bins, extra = np.full(len(base_bins), BASE_BIN), None
        else:
            bins, extra = self.policy(P, G, self.tracker, base_bins)
            act = dict(act)
            act["market"] = bins_to_orders(bins, stock, True, base_market)
        self.tracker.my_last_orders = np.array([float(b > 0) for b in (base_bins if extra is None and self.policy is None else bins)])
        if self.record:
            self.trace.append((P, G, stock, base_bins, np.asarray(bins), extra))
        return act
