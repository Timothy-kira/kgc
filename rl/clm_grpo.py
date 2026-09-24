"""Post-training: GRPO for the CLM-head decoder-only policy (MiMo-V2.6 / CodeMidas style).

Actions are slot decisions over closed candidate sets (agent/action_space.py); log-probs are softmaxes of
exp(logit_scale) * cos(state_head, action_head) over the slot's candidates, exactly as when sampling.

Per iteration
  tasks   : (seed, seat, opponent) groups, G rollouts each, all games played in parallel (rl/token_rollout.py)
  reward  : win 1 / tie 0.5 / loss 0  +  margin bonus 0.25 * tanh(diff / 15000)  (GAR-like: bigger wins rank higher)
  sampler : groups with identical rewards are dropped (no relative signal)
  adv     : r - mean(group)                        (no std normalisation, CodeMidas)
  loss    : MiMo eq.(1):  -(1/sum|o|) sum sg(ratio) * M * A * log pi   with decoupled clip bounds on the ratio,
            prompt(=trajectory)-mean aggregation; + beta * KL(pi || pi_ref) (k3 estimator) vs the mid-trained model;
            + value head (Monte-Carlo win / diff)
  masks   : log-probs are taken under the same grammar mask used when sampling (rebuilt deterministically)
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.clm_batch import collate
from model.clm_policy import CLMPolicy
from model.dsv41 import ModelArgs
from model.seq_batch import obs_feats
from rl.clm_rollout import run_games
from rl.token_rollout import WorkerPool

FIXED = ["/home/user/kgc/league/metav4.py", "/home/user/kgc/league/cha22.py", "/home/user/kgc/league/farm2945.py",
         "/home/user/kgc/league/v48.py"]
RULES = ["rule:front_run", "rule:hoarder", "rule:dripper", "rule:noisy"]


def reward_of(me, opp):
    win = 1.0 if me > opp else (0.5 if me == opp else 0.0)
    return win + 0.25 * float(np.tanh((me - opp) / 15000.0))


def token_logprobs(net, trajs, device, dtype):
    """Log-prob of every recorded decision under `net` (same candidate sets as sampling)."""
    b = collate(trajs, device=device)
    fwd = (lambda *x: net(*x)) if hasattr(net, "module") else net.forward_train
    lu, lm, value, _, aux = fwd(b["ids"], obs_feats(b, dtype), b["otype"], b["dec_pos"], b["dec_slot"], b["dec_desc"],
                                b["tgt_pos"], b["tgt_kind"], b["val_pos"])
    k, c = b["tgt_kind"], b["tgt_cand"]
    lp = torch.zeros(len(c), device=device)
    if (k == 0).any():
        lp[k == 0] = torch.log_softmax(lu.float(), -1).gather(1, c[k == 0][:, None]).squeeze(1)
    if (k == 1).any():
        lp[k == 1] = torch.log_softmax(lm.float(), -1).gather(1, c[k == 1][:, None]).squeeze(1)
    return lp, b["tgt_pos"][:, 0], value, b, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="mid-trained checkpoint (.pt)")
    ap.add_argument("--args", required=True, help="model_args.json")
    ap.add_argument("--out", default="runs/token_grpo")
    ap.add_argument("--iters", type=int, default=10 ** 6)
    ap.add_argument("--hours", type=float, default=10.5)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--clip_pos", type=float, nargs=2, default=(0.8, 1.28))
    ap.add_argument("--clip_neg", type=float, nargs=2, default=(0.72, 1.2))
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="truncate games (debug)")
    ap.add_argument("--opponents", default="fixed,rules")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    d = json.load(open(a.args))
    args = ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    net = CLMPolicy(args).to(device)
    net.load_state_dict(torch.load(a.init, map_location=device), strict=False)
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(d, open(os.path.join(a.out, "model_args.json"), "w"))
    pool = WorkerPool(a.workers)
    opp_pool = (FIXED if "fixed" in a.opponents else []) + (RULES if "rules" in a.opponents else [])
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    deadline = time.time() + a.hours * 3600
    for it in range(a.iters):
        if time.time() > deadline:
            break
        t0 = time.time()
        tasks, gid_of = [], []
        for k in range(a.tasks):
            seed, seat, opp = random.randrange(10 ** 6), random.randint(0, 1), random.choice(opp_pool)
            for g in range(a.group):
                tasks.append((seed, seat, opp))
                gid_of.append(k)
        net.eval()
        trajs = run_games(net, device, pool, tasks, temperature=1.0, max_steps=a.max_steps or None)
        rewards = np.array([reward_of(t["money_me"], t["money_opp"]) for t in trajs])
        adv = np.zeros(len(trajs))
        kept = 0
        for k in range(a.tasks):
            idx = [i for i, g in enumerate(gid_of) if g == k]
            r = rewards[idx]
            if r.std() < 1e-8:
                continue                      # dynamic sampler
            kept += 1
            adv[idx] = r - r.mean()
        use = [i for i in range(len(trajs)) if adv[i] != 0.0]
        stats = {}
        net.train()
        for ep in range(a.epochs):
            random.shuffle(use)
            for s in range(0, len(use), a.micro):
                sub = [trajs[i] for i in use[s:s + a.micro]]
                A = torch.tensor([adv[i] for i in use[s:s + a.micro]], device=device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
                    lp, tid, value, b, aux = token_logprobs(net, sub, device, dtype)
                    with torch.no_grad():
                        lp_ref, _, _, _, _ = token_logprobs(ref, sub, device, dtype)
                old = torch.tensor(np.concatenate([t["lp"] for t in sub]), device=device, dtype=torch.float32)
                ratio = torch.exp(lp - old).detach()
                At = A[tid]
                lo = torch.where(At >= 0, a.clip_pos[0], a.clip_neg[0])
                hi = torch.where(At >= 0, a.clip_pos[1], a.clip_neg[1])
                M = ((ratio >= lo) & (ratio <= hi)).float()
                n_tok = torch.bincount(tid, minlength=len(sub)).clamp(min=1).float()
                pg = -((ratio * M * At * lp) / n_tok[tid]).sum() / len(sub)
                kl = (torch.exp(lp_ref - lp) - (lp_ref - lp) - 1)
                kl = ((kl / n_tok[tid]).sum() / len(sub))
                win = torch.tensor([t["win"] for t in sub], device=device)
                diff = torch.tensor([t["diff"] for t in sub], device=device)
                bid = b["val_pos"][:, 0]
                vloss = F.binary_cross_entropy_with_logits(value[:, 0].float(), win[bid]) + \
                    0.1 * F.mse_loss(value[:, 1].float(), diff[bid])
                loss = pg + a.beta * kl + 0.1 * vloss + 0.05 * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                for k2, v in (("pg", pg), ("kl", kl), ("vloss", vloss), ("clipfrac", 1 - M.mean())):
                    stats.setdefault(k2, []).append(float(v))
        rec = {"it": it, "games": len(trajs), "kept_groups": kept, "win": float(np.mean([t["win"] for t in trajs])),
               "mean_diff": float(np.mean([t["diff"] for t in trajs])), "sec": round(time.time() - t0),
               **{k: float(np.mean(v)) for k, v in stats.items()},
               "by_opp": {o.split("/")[-1]: float(np.mean([t["win"] for t in trajs if t["opponent"] == o]))
                          for o in set(t["opponent"] for t in trajs)}}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(net.state_dict(), os.path.join(a.out, "rl_latest.pt"))
        if it % 5 == 0:
            torch.save(net.state_dict(), os.path.join(a.out, f"rl_it{it:04d}.pt"))
    pool.close()


if __name__ == "__main__":
    main()
