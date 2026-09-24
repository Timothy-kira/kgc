"""Collate trajectories (data/seq_extract.py format) into decoder-only training batches.

Sequence layout per step t:  [33 observation tokens][<ACT> a_1 ... <EOS>]
Targets: every action token after <ACT> (next-token prediction = distillation of the player's action);
value targets at each <ACT> position; MTP targets two tokens ahead within the same action.
Numpy-only here (cheap index arithmetic); tile expansion to quadrant features happens on the device.
"""
import numpy as np
import torch

from agent.obs_tokens import N_OBS, N_PROD, N_QUAD, N_UNIT, OBS_TOKEN_ID, TYPE_OF_SLOT, quad_features


def layout(traj, max_steps=None):
    T = len(traj["prod"]) if max_steps is None else min(max_steps, len(traj["prod"]))
    off = traj["act_off"][:T + 1].astype(np.int64)
    L = np.diff(off)
    base = N_OBS * np.arange(T) + off[:T]                     # sequence start of each step
    S = int(N_OBS * T + off[T])
    ids = np.empty(S, np.int64)
    otype = np.full(S, -1, np.int8)
    obs_pos = (base[:, None] + np.arange(N_OBS)[None]).ravel()
    ids[obs_pos] = np.tile(np.array(OBS_TOKEN_ID)[TYPE_OF_SLOT], T)
    otype[obs_pos] = np.tile(TYPE_OF_SLOT, T)
    act_start = base + N_OBS
    act_pos = np.repeat(act_start - off[:T], L) + np.arange(off[T])   # position of each action token
    ids[act_pos] = traj["act"][:off[T]]
    # next-token targets inside each action (skip the step boundary)
    step_of = np.repeat(np.arange(T), L)
    first = np.zeros(off[T], bool)
    first[off[:T]] = True                                      # <ACT> tokens
    nxt_ok = np.ones(off[T], bool)
    nxt_ok[off[1:T + 1] - 1] = False                           # last token (<EOS>) of each action
    tgt_src = act_pos[nxt_ok]
    tgt_id = ids[tgt_src + 1]
    mtp_ok = nxt_ok.copy()
    mtp_ok[np.maximum(off[1:T + 1] - 2, 0)] = False
    mtp_src = act_pos[mtp_ok]
    mtp_id = ids[mtp_src + 2]
    return dict(S=S, T=T, ids=ids, otype=otype, tgt_src=tgt_src, tgt_id=tgt_id, val_src=act_pos[first],
                mtp_src=mtp_src, mtp_id=mtp_id, step_of_tgt=step_of[nxt_ok])


def collate(trajs, device="cpu", max_steps=None, value_every=4):
    lays = [layout(t, max_steps) for t in trajs]
    B = len(trajs)
    S = max(l["S"] for l in lays)
    ids = np.zeros((B, S), np.int64)
    otype = np.full((B, S), -1, np.int8)
    tp, ti, tw, vp, vt, mp_, mi = [], [], [], [], [], [], []
    prod, glob, tiles, units = [], [], [], []
    for b, (tr, l) in enumerate(zip(trajs, lays)):
        ids[b, :l["S"]] = l["ids"]
        otype[b, :l["S"]] = l["otype"]
        w = (1.0 if tr["win"] > 0.5 else 0.5) * min(1.5, max(0.3, (tr["score"] - 1500) / 1500))
        tp.append(np.stack([np.full(len(l["tgt_src"]), b), l["tgt_src"]], 1))
        ti.append(l["tgt_id"])
        tw.append(np.full(len(l["tgt_src"]), w, np.float32))
        v = l["val_src"][::value_every]
        vp.append(np.stack([np.full(len(v), b), v], 1))
        vt.append(np.tile([[tr["win"], tr["diff"]]], (len(v), 1)).astype(np.float32))
        mp_.append(np.stack([np.full(len(l["mtp_src"]), b), l["mtp_src"]], 1))
        mi.append(l["mtp_id"])
        T = l["T"]
        prod.append(tr["prod"][:T].reshape(-1, tr["prod"].shape[-1]))
        glob.append(tr["glob"][:T])
        tiles.append(tr["tiles"][:T])
        units.append(tr["units"][:T].reshape(-1, tr["units"].shape[-1]))
    t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True) if dt is None \
        else torch.from_numpy(np.ascontiguousarray(x)).to(device, dtype=dt, non_blocking=True)
    return dict(ids=t(ids), otype=t(otype), tgt_pos=t(np.concatenate(tp)), tgt_ids=t(np.concatenate(ti).astype(np.int64)),
                tgt_w=t(np.concatenate(tw)), val_pos=t(np.concatenate(vp)), val_tgt=t(np.concatenate(vt)),
                mtp_pos=t(np.concatenate(mp_)), mtp_ids=t(np.concatenate(mi).astype(np.int64)),
                prod=t(np.concatenate(prod)), glob=t(np.concatenate(glob)), tiles=t(np.concatenate(tiles)),
                units=t(np.concatenate(units)))


def obs_feats(batch, dtype):
    """Expand stored observation arrays into per-type feature rows (on the batch's device)."""
    quads = quad_features(batch["tiles"].to(dtype) / 255.0).reshape(-1, quad_features(
        np.zeros((1, 2, 100, batch["tiles"].shape[-1]), np.float32)).shape[-1])
    return [batch["prod"].to(dtype), batch["glob"].to(dtype), quads, batch["units"].to(dtype)]
