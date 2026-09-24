"""Check that the numpy inference twin reproduces the torch training model exactly on a real trace."""
import numpy as np
import torch

from model.net import TTTPolicy
from model.policy_np import NumpyPolicy
from rl.rollout import rollout


def main():
    torch.manual_seed(0)
    net = TTTPolicy()
    with torch.no_grad():  # make TTT path non-trivial
        net.w_fast0.normal_(std=0.05)
        net.log_eta.fill_(0.0)
    net.export("/tmp/eq_w.npz")
    tr = rollout({"weights": "/tmp/eq_w.npz", "opponent": "/home/user/kgc/league/cha22.py", "seed": 7, "seat": 0,
                  "greedy": True, "rng_seed": 0})
    T = tr["T"]
    day_idx = torch.arange(T) // 24
    with torch.no_grad():
        logits, value, ssl = net(torch.tensor(tr["P"])[None], torch.tensor(tr["G"])[None], day_idx,
                                 torch.tensor(tr["day_tgt"])[None], torch.tensor(tr["day_mask"])[None])
    # recompute numpy logits along the same inputs
    pol = NumpyPolicy("/tmp/eq_w.npz")

    class Tr:
        def __init__(self):
            self.q = []

        def pop_day_targets(self):
            q, self.q = self.q, []
            return q
    fake = Tr()
    tg = {d: tr["day_tgt"][d] for d in range(len(tr["day_mask"])) if tr["day_mask"][d] > 0}
    maxdiff = 0.0
    for t in range(T):
        if t % 24 == 0 and t > 0 and (t // 24 - 1) in tg:
            fake.q.append((t // 24 - 1, tg[t // 24 - 1]))
        lg, v = pol.forward(tr["P"][t], tr["G"][t], fake)
        maxdiff = max(maxdiff, float(np.abs(lg - logits[0, t].numpy()).max()))
    print("T", T, "max |logit diff| numpy vs torch:", maxdiff, "ssl", float(ssl), "money", tr["money_me"], tr["money_opp"])


if __name__ == "__main__":
    main()
