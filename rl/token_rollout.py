"""Parallel game rollouts for the decoder-only token policy (post-training RL).

Many games run at once. Our side is decoded in one batch on the GPU (`Transformer.step_multi`, each game at
its own sequence position); environments + opponents live in CPU worker processes.

Every rollout returns a trajectory in exactly the replay-extraction format (data/seq_extract.py), plus the
sampled tokens' log-probs, so the RL update can reuse `model.seq_batch.collate` + `forward_train`.

Idle games (their action already finished while others are still decoding) are fed a dummy token at their
*current, not yet used* position without advancing it; every cache slot written that way (window ring,
compressor state, compressed KV at that group) is overwritten by the real token later, so it is harmless.
"""
import multiprocessing as mp
import time

import numpy as np
import torch

from agent.action_tokens import TOK, Grammar, decode
from agent.features import Tracker
from agent.obs_tokens import N_OBS, OBS_TOKEN_ID, TYPE_OF_SLOT, build_step, quad_features
from env.fast_env import FarmEnv

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


# ------------------------------------------------------------------------------------------ CPU workers
def _make_opponent(spec, seed):
    rng = np.random.default_rng(seed)
    if spec.startswith("rule:"):
        from rl.rule_adversaries import make_rule_adversary
        from tools.arena import load
        return make_rule_adversary(spec[5:], load("/home/user/kgc/league/metav4.py"), rng)
    if spec.startswith("seq:"):
        raise ValueError("self-play snapshots are played on the GPU side, not in workers")
    from tools.arena import load
    return load(spec)


def _worker(conn):
    games = {}
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "new":
            _, gid, seed, seat, opp_spec = msg
            env = FarmEnv(seed)
            games[gid] = dict(env=env, seat=seat, opp=_make_opponent(opp_spec, seed), trk=Tracker())
            conn.send(("ok", gid))
        elif cmd == "obs":                                  # our observation arrays for this step
            out = {}
            for gid in msg[1]:
                g = games[gid]
                o = g["env"].obs(g["seat"])
                g["obs"] = o
                g["trk"].update(o)
                P, G, _, _ = g["trk"].features(None)
                st = build_step(o, P, G)
                st["n_hands"] = len(o["farms"][g["seat"]]["hands"])
                out[gid] = st
            conn.send(out)
        elif cmd == "act":                                  # our actions -> step envs with opponent moves
            res = {}
            for gid, a in msg[1].items():
                g = games[gid]
                env, seat = g["env"], g["seat"]
                oa = g["opp"](env.obs(1 - seat))
                env.step(*((a, oa) if seat == 0 else (oa, a)))
                res[gid] = (env.done, env.money[seat], env.money[1 - seat])
            conn.send(res)
        elif cmd == "close":
            for gid in msg[1]:
                games.pop(gid, None)
            conn.send({})
        elif cmd == "exit":
            return


class WorkerPool:
    def __init__(self, n):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def owner(self, gid):
        return self.conns[gid % len(self.conns)]

    def broadcast(self, cmd, per_gid):
        """per_gid: {gid: payload}; sends grouped by worker, returns merged replies."""
        groups = {}
        for gid, v in per_gid.items():
            groups.setdefault(gid % len(self.conns), {})[gid] = v
        for w, d in groups.items():
            self.conns[w].send((cmd, d if cmd == "act" else list(d)))
        out = {}
        for w in groups:
            out.update(self.conns[w].recv())
        return out

    def new_game(self, gid, seed, seat, opp):
        c = self.owner(gid)
        c.send(("new", gid, seed, seat, opp))
        c.recv()

    def close(self):
        for c in self.conns:
            c.send(("exit",))


# ------------------------------------------------------------------------------------------ batched policy
class BatchedPolicy:
    def __init__(self, net, device, temperature=1.0, max_tokens=90):
        self.net, self.device, self.temperature, self.max_tokens = net, device, temperature, max_tokens

    @torch.no_grad()
    def obs_embeddings(self, sts):
        """list of step dicts -> [B, N_OBS, dim] embeddings (observation tokens)."""
        dev, net = self.device, self.net
        prod = torch.from_numpy(np.stack([s["prod"] for s in sts]).astype(np.float32)).to(dev)
        glob = torch.from_numpy(np.stack([s["glob"] for s in sts]).astype(np.float32)).to(dev)
        tiles = torch.from_numpy(np.stack([s["tiles"] for s in sts])).to(dev).float() / 255.0
        units = torch.from_numpy(np.stack([s["units"] for s in sts]).astype(np.float32)).to(dev)
        quads = quad_features(tiles)                                             # [B,8,QUAD_W]
        ids = torch.tensor([OBS_TOKEN_ID[t] for t in TYPE_OF_SLOT], device=dev)
        h = net.embed(ids).unsqueeze(0).expand(len(sts), -1, -1).clone()
        feats = [prod, glob.unsqueeze(1), quads, units]
        start = 0
        for t, f in enumerate(feats):
            n = f.size(1)
            h[:, start:start + n] += net.obs_proj[t](f)
            start += n
        return h

    @torch.no_grad()
    def act(self, sts, caches, pos):
        """Decode one action for every game. pos: LongTensor [B] (updated in place)."""
        net, dev = self.net, self.device
        B = len(sts)
        h_obs = self.obs_embeddings(sts)
        act_tok = net.embed(torch.full((B, 1), TOK["<ACT>"], device=dev))
        seq = torch.cat([h_obs, act_tok], 1)
        h = None
        for i in range(seq.size(1)):                          # observation block + <ACT>, all games in lockstep
            h = net.step_multi(seq[:, i:i + 1], pos, caches)
            pos += 1
        grams = [Grammar(max_hands=s["n_hands"], allow_base=False) for s in sts]
        for g in grams:
            g.feed(TOK["<ACT>"])
        toks = [[TOK["<ACT>"]] for _ in range(B)]
        lps = [[] for _ in range(B)]
        active = torch.ones(B, dtype=torch.bool, device=dev)
        V = net.head.out_features
        for it in range(self.max_tokens):
            logits = net.head(h[:, 0].float())                                   # [B,V]
            mask = np.zeros((B, V), bool)
            for b in range(B):
                if active[b]:
                    m = grams[b].mask()
                    mask[b, :len(m)] = m
                else:
                    mask[b, TOK["<PAD>"]] = True
            mask_t = torch.from_numpy(mask).to(dev)
            logits = logits.masked_fill(~mask_t, -1e9)
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, -1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
                lp = torch.log(probs.gather(1, nxt[:, None]).squeeze(1) + 1e-12)
            else:
                nxt = logits.argmax(-1)
                lp = torch.zeros(B, device=dev)
            nxt_c, lp_c = nxt.tolist(), lp.tolist()
            for b in range(B):
                if active[b]:
                    grams[b].feed(nxt_c[b])
                    toks[b].append(nxt_c[b])
                    lps[b].append(lp_c[b])
            was_active = active.clone()
            active = torch.tensor([not grams[b].done for b in range(B)], device=dev)
            # feed the sampled token (dummy <PAD> for idle games at their unused current position)
            feed = torch.where(was_active, nxt, torch.full_like(nxt, TOK["<PAD>"]))
            h = net.step_multi(net.embed(feed[:, None]), pos, caches)
            pos += was_active.long()
            if not bool(active.any()):
                break
        acts = []
        for b in range(B):
            try:
                a = decode(toks[b]) if grams[b].done else PASS
            except Exception:
                a = PASS
            acts.append(PASS if a == "BASE" else a)
        return acts, toks, lps


def run_games(net, device, pool, tasks, temperature=1.0, max_steps=None):
    """tasks: list of (seed, seat, opponent_spec). Returns trajectories (replay-extraction format + logps)."""
    B = len(tasks)
    for gid, (seed, seat, opp) in enumerate(tasks):
        pool.new_game(gid, seed, seat, opp)
    caches = net.stack_caches([net.init_cache(1, device) for _ in range(B)])
    pos = torch.zeros(B, dtype=torch.long, device=device)
    pol = BatchedPolicy(net, device, temperature)
    rec = [dict(prod=[], glob=[], tiles=[], units=[], act=[], off=[0], lp=[]) for _ in range(B)]
    alive = list(range(B))
    result = {}
    t0 = time.time()
    step = 0
    while alive:
        sts = pool.broadcast("obs", {g: None for g in alive})
        st_list = [sts[g] for g in alive]
        if len(alive) < B:          # finished games keep their slot: feed them a copy (never recorded)
            filler = st_list[0]
            full = [sts.get(g, filler) for g in range(B)]
        else:
            full = st_list
        acts, toks, lps = pol.act(full, caches, pos)
        for g in alive:
            s, R = sts[g], rec[g]
            R["prod"].append(s["prod"]); R["glob"].append(s["glob"]); R["tiles"].append(s["tiles"])
            R["units"].append(s["units"]); R["act"].extend(toks[g]); R["off"].append(len(R["act"]))
            R["lp"].append(lps[g])
        res = pool.broadcast("act", {g: acts[g] for g in alive})
        for g, (done, mm, mo) in res.items():
            if done or (max_steps and step + 1 >= max_steps):
                result[g] = (mm, mo)
        alive = [g for g in alive if g not in result]
        step += 1
    pool.broadcast("close", {g: None for g in range(B)})
    trajs = []
    for g, (seed, seat, opp) in enumerate(tasks):
        R = rec[g]
        mm, mo = result[g]
        trajs.append(dict(prod=np.array(R["prod"]), glob=np.array(R["glob"]), tiles=np.array(R["tiles"]),
                          units=np.array(R["units"]), act=np.array(R["act"], np.int16),
                          act_off=np.array(R["off"], np.int32), lp=R["lp"],
                          win=float(1.0 if mm > mo else 0.5 if mm == mo else 0.0), diff=float((mm - mo) / 1e4),
                          score=3000.0, money_me=mm, money_opp=mo, seed=seed, seat=seat, opponent=opp))
    print(f"[rollout] {B} games x {step} steps in {time.time() - t0:.0f}s", flush=True)
    return trajs
