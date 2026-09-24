"""Parallel game rollouts for the CLM-head policy (post-training RL).

Same structure as rl/token_rollout.py: CPU worker processes own the environments + opponents; our side is
decided in one batch on the GPU with `step_multi` (every game at its own sequence position). Each step:
observation block + <ACT> fed token by token (all games in lockstep), then slot decisions scored with the CLM
heads against the cached candidate matrices. Games whose action is already complete are fed a dummy at their
current, unused position (overwritten later), exactly as in token_rollout.

Trajectories are returned in the replay format plus `dec_slot / dec_cand / dec_off` (so clm_batch needs no
token decoding) and per-decision log-probs.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.action_space import SLOT_MARKET, SlotPlan, decisions_to_action
from agent.obs_tokens import OBS_TOKEN_ID, TYPE_OF_SLOT, quad_features
from model.clm_policy import ACT_ID, OBS_IDS, SLOT_IDS
from rl.token_rollout import WorkerPool  # noqa: F401  (same CPU workers)


class CLMBatchedPolicy:
    def __init__(self, net, device, temperature=1.0):
        self.net, self.device, self.temperature = net, device, temperature
        with torch.no_grad():
            self.zu, self.zm = net.candidate_matrices()
            self.enc_u = net.action_enc(net.unit_desc)
            self.enc_m = net.action_enc(net.market_desc)
            self.slot_emb = net.embed(torch.tensor(SLOT_IDS, device=device))

    @torch.no_grad()
    def obs_embeddings(self, sts):
        dev, net = self.device, self.net
        prod = torch.from_numpy(np.stack([s["prod"] for s in sts]).astype(np.float32)).to(dev)
        glob = torch.from_numpy(np.stack([s["glob"] for s in sts]).astype(np.float32)).to(dev)
        tiles = torch.from_numpy(np.stack([s["tiles"] for s in sts])).to(dev).float() / 255.0
        units = torch.from_numpy(np.stack([s["units"] for s in sts]).astype(np.float32)).to(dev)
        ids = torch.tensor([OBS_IDS[t] for t in TYPE_OF_SLOT], device=dev)
        h = net.embed(ids).unsqueeze(0).expand(len(sts), -1, -1).clone()
        feats = [prod, glob.unsqueeze(1), quad_features(tiles), units]
        start = 0
        for t, f in enumerate(feats):
            n = f.size(1)
            h[:, start:start + n] += net.obs_proj[t](f)
            start += n
        return h

    @torch.no_grad()
    def act(self, sts, caches, pos):
        net, dev = self.net, self.device
        B = len(sts)
        seq = torch.cat([self.obs_embeddings(sts), net.embed(torch.full((B, 1), ACT_ID, device=dev))], 1)
        h = None
        for i in range(seq.size(1)):
            h = net.step_multi(seq[:, i:i + 1], pos, caches)
            pos += 1
        plans = [SlotPlan(s["n_hands"]) for s in sts]
        decs = [[] for _ in range(B)]
        lps = [[] for _ in range(B)]
        scale = net.scale()
        while True:
            active = [b for b in range(B) if not plans[b].done]
            if not active:
                break
            zs = F.normalize(net.state_head(h[:, 0]).float(), dim=-1)            # [B,P]
            feed = torch.zeros(B, net.args.dim, device=dev)
            adv = torch.zeros(B, dtype=torch.long, device=dev)
            slots = [plans[b].next_slot() if b in active else None for b in range(B)]
            mk = [b for b in active if slots[b] == SLOT_MARKET]
            uk = [b for b in active if slots[b] != SLOT_MARKET]
            for group, Z, enc in ((uk, self.zu, self.enc_u), (mk, self.zm, self.enc_m)):
                if not group:
                    continue
                gi = torch.tensor(group, device=dev)
                lg = scale * zs[gi] @ Z.t()
                if self.temperature > 0:
                    p = torch.softmax(lg / self.temperature, -1)
                    c = torch.multinomial(p, 1).squeeze(1)
                    lp = torch.log(p.gather(1, c[:, None]).squeeze(1) + 1e-12)
                else:
                    c = lg.argmax(-1)
                    lp = torch.zeros(len(group), device=dev)
                for j, b in enumerate(group):
                    cb = int(c[j])
                    plans[b].feed(slots[b], cb)
                    decs[b].append((slots[b], cb))
                    lps[b].append(float(lp[j]))
                    feed[b] = self.slot_emb[min(slots[b], 2)] + enc[cb]
                    adv[b] = 1
            h = net.step_multi(feed[:, None].to(self.slot_emb.dtype), pos, caches)
            pos += adv
        return [decisions_to_action(d) for d in decs], decs, lps


def run_games(net, device, pool, tasks, temperature=1.0, max_steps=None):
    """tasks: list of (seed, seat, opponent_spec) -> trajectories."""
    B = len(tasks)
    for gid, (seed, seat, opp) in enumerate(tasks):
        pool.new_game(gid, seed, seat, opp)
    caches = net.stack_caches([net.init_cache(1, device) for _ in range(B)])
    pos = torch.zeros(B, dtype=torch.long, device=device)
    pol = CLMBatchedPolicy(net, device, temperature)
    rec = [dict(prod=[], glob=[], tiles=[], units=[], dec_slot=[], dec_cand=[], dec_off=[0], lp=[]) for _ in range(B)]
    alive, result, step, t0 = list(range(B)), {}, 0, time.time()
    while alive:
        sts = pool.broadcast("obs", {g: None for g in alive})
        full = [sts.get(g, sts[alive[0]]) for g in range(B)]
        acts, decs, lps = pol.act(full, caches, pos)
        for g in alive:
            s, R = sts[g], rec[g]
            for k in ("prod", "glob", "tiles", "units"):
                R[k].append(s[k])
            R["dec_slot"] += [d[0] for d in decs[g]]
            R["dec_cand"] += [d[1] for d in decs[g]]
            R["dec_off"].append(len(R["dec_slot"]))
            R["lp"] += lps[g]
        res = pool.broadcast("act", {g: acts[g] for g in alive})
        for g, (done, mm, mo) in res.items():
            if done or (max_steps and step + 1 >= max_steps):
                result[g] = (mm, mo)
        alive = [g for g in alive if g not in result]
        step += 1
    pool.broadcast("close", {g: None for g in range(B)})
    trajs = []
    for g, (seed, seat, opp) in enumerate(tasks):
        R, (mm, mo) = rec[g], result[g]
        trajs.append(dict(prod=np.array(R["prod"]), glob=np.array(R["glob"]), tiles=np.array(R["tiles"]),
                          units=np.array(R["units"]), dec_slot=np.array(R["dec_slot"], np.int8),
                          dec_cand=np.array(R["dec_cand"], np.int16), dec_off=np.array(R["dec_off"], np.int32),
                          lp=np.array(R["lp"], np.float32), act=np.zeros(0, np.int16), act_off=np.zeros(1, np.int32),
                          win=float(1.0 if mm > mo else 0.5 if mm == mo else 0.0), diff=float((mm - mo) / 1e4),
                          score=3000.0, money_me=mm, money_opp=mo, seed=seed, seat=seat, opponent=opp))
    print(f"[rollout] {B} games x {step} steps in {time.time() - t0:.0f}s", flush=True)
    return trajs
