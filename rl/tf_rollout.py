"""B-track rollouts (docs/PLAN_RL.md): the pure-Transformer policy (CSA2 decoder + official Engram + CLM heads)
decides every slot; cha22/v4b is only a training-time teacher and opponent.

* CPU workers own the environments, the opponents (rl/opp_mix.py), the feature trackers and each game's live Engram
  state (agent/tf_agent.EngramLive: inferred opponent flows -> S/D event streams -> hash rows, identical to training).
* run_games(): all games in lockstep; our side is decoded in one batch on the GPU (rl/clm_rollout.CLMBatchedPolicy)
  after setting the step's Engram rows, so every token of step t sees the S/D n-grams visible at t.
* teacher_game(): a whole game played by the teacher (v4b) on the CPU, recorded in the same trajectory format with
  the teacher's actions as slot decisions (on-the-fly distillation data, no GPU needed).
Trajectories are model/clm_batch.collate-ready: prod/glob/tiles/units, dec_slot/dec_cand/dec_off, eng_s/eng_d,
opp_tgt/opp_stock, win/diff/score (+ lp for sampled decisions).
"""
import importlib
import json
import multiprocessing as mp
import random
import time
import zlib

import numpy as np
import torch

from agent.action_space import action_to_decisions
from agent.features import Tracker
from agent.obs_tokens import build_step
from agent.opp_events import PRODUCTS, infer_flows, opp_targets
from agent.tf_agent import EngramLive
from env.fast_env import FarmEnv
from model.opp_data import LAYER_D, LAYER_S
from rl.clm_rollout import CLMBatchedPolicy
from rl.opp_mix import OppMix, SeatTape, Tape1, v4b

K = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
W = {}                                                   # worker globals: vocab, mix (set before forking)


class View:
    """One seat's view of a game: feature tracker, live Engram state, inferred opponent flows, hash rows."""

    def __init__(self, seat, vocab):
        self.seat = seat
        self.trk, self.live = Tracker(), EngramLive(vocab)
        self.prev_obs = self.prev_action = None
        self.flow, self.rows_s, self.rows_d = [], [], []
        self.o = None

    def sync(self, o):
        if self.prev_obs is not None:
            try:
                fl = infer_flows(K, self.prev_obs, self.prev_action, o["market"]["inventory"])
            except Exception:
                fl = {}
            f = o["farms"][1 - self.seat]
            self.live.push(int(self.prev_obs["step"]), fl, f, f["money"], o["market"]["inventory"]["WHEAT"])
            self.flow.append([int(fl.get(p, 0)) for p in PRODUCTS])

    def observe(self, env):
        o = env.obs(self.seat)
        self.sync(o)
        self.o = o
        self.trk.update(o)
        P, G, _, _ = self.trk.features(None)
        st = build_step(o, P, G)
        st["n_hands"] = len(o["farms"][self.seat]["hands"])
        rs, rd = self.live.rows()
        self.rows_s.append(rs)
        self.rows_d.append(rd)
        st["rows_s"], st["rows_d"] = rs, rd
        return st

    def after(self, action):
        self.prev_obs, self.prev_action = self.o, action


class Game:
    """One game from our seat's view. The opponent is a program / tape (rl/opp_mix.py), a given callable
    (how = ("callable", f)), or - self-play, how = ("self", snapshot) - a second View decoded on the GPU."""

    def __init__(self, spec, vocab, teacher=None):
        kind, name, seed, seat, cfg, how = spec
        self.kind, self.name, self.seat = kind, name, seat
        self.env = FarmEnv(seed, cfg)
        self.me = View(seat, vocab)
        self.opp_view = View(1 - seat, vocab) if how[0] == "self" else None
        self.opp = None if self.opp_view is not None else (how[1] if how[0] == "callable" else OppMix.make(how))
        self.teacher = teacher
        self.shed = []

    @property
    def o(self):
        return self.me.o

    def observe(self):
        st = self.me.observe(self.env)
        if self.opp_view is not None:
            st["opp_st"] = self.opp_view.observe(self.env)
        return st

    def act(self, action, opp_action=None):
        env, seat = self.env, self.seat
        if self.opp_view is not None:
            oa = opp_action
            self.opp_view.after(oa)
        else:
            tape = isinstance(self.opp, Tape1)             # open-loop tapes only read obs["step"]: no copy
            oa = self.opp(env.obs(1 - seat, copy_obs=not tape), env.config)
        env.step(*((action, oa) if seat == 0 else (oa, action)))
        self.me.after(action)
        sh = env.obs(1 - seat, copy_obs=False)["private"]["shed"]
        self.shed.append([int(sh.get(p, 0)) for p in PRODUCTS])
        if env.done:
            self.me.sync(env.obs(seat))                    # flows of the final transition (opponent targets)
        return env.done, env.money[seat], env.money[1 - seat]

    def targets(self, T):
        m = self.me
        flow = np.array(m.flow[:T] + [[0] * len(PRODUCTS)] * max(0, T - len(m.flow)), np.int64)
        shed = np.array(self.shed[:T], np.float32).reshape(-1, len(PRODUCTS))
        before = np.concatenate([np.zeros((1, len(PRODUCTS)), np.float32), shed])[:T]
        return dict(eng_s=np.stack(m.rows_s[:T]), eng_d=np.stack(m.rows_d[:T]), opp_tgt=opp_targets(flow),
                    opp_stock=np.log1p(before))


def pack(steps, decs, extra):
    """steps: list of build_step dicts; decs: list (per step) of [(slot, cand)] -> trajectory dict."""
    tr = {k: np.stack([s[k] for s in steps]) for k in ("prod", "glob", "tiles", "units")}
    off = np.concatenate([[0], np.cumsum([len(d) for d in decs])]).astype(np.int32)
    flat = [x for d in decs for x in d]
    tr["dec_slot"] = np.array([s for s, _ in flat], np.int8)
    tr["dec_cand"] = np.array([c for _, c in flat], np.int16)
    tr["dec_off"] = off
    tr.update(extra)
    return tr


def outcome(mm, mo):
    return dict(win=float(1.0 if mm > mo else 0.5 if mm == mo else 0.0), diff=float((mm - mo) / 1e4), score=3000.0,
                money_me=float(mm), money_opp=float(mo))


def teacher_game(j):
    """job -> distillation trajectory from our seat: with prob W["demo_frac"] a replayed high-rated winner
    (rl/opp_mix.export_v2 "demo": both seats replayed, decisions = the winner's recorded action), otherwise v4b vs
    W["mix"].spec(j) (decisions = v4b's action)."""
    torch.set_num_threads(1)
    rng = random.Random(j)
    demo = W.get("demo")
    if demo and rng.random() < W.get("demo_frac", 0.0):
        eid, seed, cfg, blob, w = demo[j % len(demo)]
        acts = json.loads(zlib.decompress(blob))
        spec = ("demo", f"demo:{eid}", seed, w, cfg, ("callable", SeatTape(acts, 1 - w)))
        g = Game(spec, W["vocab"], teacher=SeatTape(acts, w))
    else:
        g = Game(W["mix"].spec(j), W["vocab"], teacher=v4b())
    steps, decs = [], []
    done, mm, mo = False, 0, 0
    while not done:
        st = g.observe()
        a = g.teacher(g.o, g.env.config)
        steps.append(st)
        decs.append(action_to_decisions(a, st["n_hands"]))
        done, mm, mo = g.act(a)
    T = len(steps)
    tr = pack(steps, decs, g.targets(T))
    tr.update(outcome(mm, mo), kind=g.kind, opponent=g.name, job=j)
    return tr


# ------------------------------------------------------------------------------------------ student rollouts
def _worker(conn, vocab):
    games = {}
    torch.set_num_threads(1)
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "new":
            for gid, spec in msg[1].items():
                games[gid] = Game(spec, vocab)
            conn.send({})
        elif cmd == "obs":
            conn.send({gid: games[gid].observe() for gid in msg[1]})
        elif cmd == "act":
            conn.send({gid: games[gid].act(*a) for gid, a in msg[1].items()})
        elif cmd == "targets":
            conn.send({gid: games[gid].targets(T) for gid, T in msg[1].items()})
        elif cmd == "close":
            for gid in msg[1]:
                games.pop(gid, None)
            conn.send({})
        elif cmd == "exit":
            return


class Workers:
    def __init__(self, n, vocab):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, vocab), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def call(self, cmd, per_gid):
        groups = {}
        for gid, v in per_gid.items():
            groups.setdefault(gid % len(self.conns), {})[gid] = v
        for w, d in groups.items():
            self.conns[w].send((cmd, d))
        out = {}
        for w in groups:
            out.update(self.conns[w].recv())
        return out

    def close(self):
        for c in self.conns:
            c.send(("exit",))


def run_games(net, device, workers, specs, temperature=1.0, max_steps=None, opp_net=None):
    """specs: list of opponent specs (rl/opp_mix.OppMix.spec, or how = ("self", tag) for self-play against
    `opp_net`, decoded greedily in its own batch) -> student trajectories (+ per-decision lp)."""
    B = len(specs)
    workers.call("new", {g: s for g, s in enumerate(specs)})
    caches = net.stack_caches([net.init_cache(1, device) for _ in range(B)])
    pos = torch.zeros(B, dtype=torch.long, device=device)
    pol = CLMBatchedPolicy(net, device, temperature)
    sp = [g for g, s in enumerate(specs) if s[5][0] == "self"]
    if sp:
        caches_o = opp_net.stack_caches([opp_net.init_cache(1, device) for _ in sp])
        pos_o = torch.zeros(len(sp), dtype=torch.long, device=device)
        pol_o = CLMBatchedPolicy(opp_net, device, 0.0)
    steps = [[] for _ in range(B)]
    decs = [[] for _ in range(B)]
    lps = [[] for _ in range(B)]
    alive, result, n, t0 = list(range(B)), {}, 0, time.time()

    def rows(m, sts):
        rs = torch.from_numpy(np.stack([s["rows_s"] for s in sts])).to(device).unsqueeze(1)
        rd = torch.from_numpy(np.stack([s["rows_d"] for s in sts])).to(device).unsqueeze(1)
        m.engram_ids, m.engram_mask = {LAYER_S: rs, LAYER_D: rd}, None

    while alive:
        sts = workers.call("obs", {g: None for g in alive})
        full = [sts.get(g, sts[alive[0]]) for g in range(B)]
        rows(net, full)
        acts, dd, ll = pol.act(full, caches, pos)
        opp_act = {}
        live_sp = [g for g in sp if g in sts]
        if live_sp:
            fo = [sts[g]["opp_st"] if g in sts else sts[live_sp[0]]["opp_st"] for g in sp]
            rows(opp_net, fo)
            ao, _, _ = pol_o.act(fo, caches_o, pos_o)
            opp_act = {g: ao[i] for i, g in enumerate(sp)}
        for g in alive:
            steps[g].append({k: sts[g][k] for k in ("prod", "glob", "tiles", "units")})
            decs[g].append(dd[g])
            lps[g] += ll[g]
        res = workers.call("act", {g: (acts[g], opp_act.get(g)) for g in alive})
        for g, (done, mm, mo) in res.items():
            if done or (max_steps and n + 1 >= max_steps):
                result[g] = (mm, mo)
        alive = [g for g in alive if g not in result]
        n += 1
    net.engram_ids = None
    if sp:
        opp_net.engram_ids = None
    tg = workers.call("targets", {g: len(steps[g]) for g in range(B)})
    workers.call("close", {g: None for g in range(B)})
    trajs = []
    for g, spec in enumerate(specs):
        tr = pack(steps[g], decs[g], tg[g])
        tr.update(outcome(*result[g]), lp=np.array(lps[g], np.float32), kind=spec[0], opponent=spec[1])
        trajs.append(tr)
    print(f"[rollout] {B} games x {n} steps in {time.time() - t0:.0f}s", flush=True)
    return trajs


def base_game(args):
    """(job, spec) -> (job, kind, diff) of v4b itself on that job (paired baseline for EVAL)."""
    j, spec = args
    torch.set_num_threads(1)
    kind, name, seed, seat, cfg, how = spec
    me, opp = v4b(), OppMix.make(how)
    env = FarmEnv(seed, cfg)
    while not env.done:
        o = [env.obs(0), env.obs(1)]
        a, b = me(o[seat], env.config), opp(o[1 - seat], env.config)
        env.step(*((a, b) if seat == 0 else (b, a)))
    return j, kind, env.money[seat] - env.money[1 - seat]
