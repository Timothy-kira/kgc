"""Collate trajectories into batches for the CLM-head policy.

Trajectories come from data/seq_extract.py (action tokens); each step's tokens are decoded back into the
action and re-expressed as slot decisions (agent/action_space.py). Layout per step:
    [33 obs][<ACT>][d_1]...[d_m]      decision j predicted from the position before it.
"""
import numpy as np
import torch

from agent.action_space import MARKET_DESC, SLOT_MARKET, UNIT_DESC, action_to_decisions
from agent.action_tokens import decode
from agent.obs_tokens import N_OBS, TYPE_OF_SLOT
from model.clm_policy import ACT_ID, OBS_IDS, SLOT_IDS

SLOT_TOKEN = np.array(SLOT_IDS)


def trajectory_decisions(tr):
    """-> dec_slot int8 [D], dec_cand int16 [D], dec_off int32 [T+1] (cached on the trajectory)."""
    if "dec_off" in tr:
        return tr["dec_slot"], tr["dec_cand"], tr["dec_off"]
    n_hands = (tr["units"][:, :, 34] > 0.5).sum(1) - 1
    off = tr["act_off"]
    slots, cands, doff = [], [], [0]
    for t in range(len(off) - 1):
        toks = [int(x) for x in tr["act"][off[t]:off[t + 1]]]
        try:
            a = decode(toks)
        except Exception:
            a = None
        a = a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []}
        for s, c in action_to_decisions(a, int(n_hands[t])):
            slots.append(s)
            cands.append(c)
        doff.append(len(slots))
    tr["dec_slot"] = np.array(slots, np.int8)
    tr["dec_cand"] = np.array(cands, np.int16)
    tr["dec_off"] = np.array(doff, np.int32)
    return tr["dec_slot"], tr["dec_cand"], tr["dec_off"]


def layout(tr, max_steps=None):
    ds, dc, doff = trajectory_decisions(tr)
    T = len(tr["prod"]) if max_steps is None else min(max_steps, len(tr["prod"]))
    doff = doff[:T + 1].astype(np.int64)
    L = np.diff(doff)                               # decisions per step
    per = N_OBS + 1 + L                             # tokens per step
    base = np.concatenate([[0], np.cumsum(per)[:-1]])
    S = int(per.sum())
    ids = np.empty(S, np.int64)
    otype = np.full(S, -1, np.int8)
    obs_pos = (base[:, None] + np.arange(N_OBS)[None]).ravel()
    ids[obs_pos] = np.tile(np.array(OBS_IDS)[TYPE_OF_SLOT], T)
    otype[obs_pos] = np.tile(TYPE_OF_SLOT, T)
    act_pos = base + N_OBS
    ids[act_pos] = ACT_ID
    nd = int(doff[T])
    dslot = ds[:nd].astype(np.int64)
    dcand = dc[:nd].astype(np.int64)
    dec_pos = np.repeat(act_pos + 1 - doff[:T], L) + np.arange(nd)
    ids[dec_pos] = SLOT_TOKEN[dslot]
    kind = (dslot == SLOT_MARKET).astype(np.int64)
    desc = np.where(kind[:, None] == 1, MARKET_DESC[np.minimum(dcand, len(MARKET_DESC) - 1)],
                    UNIT_DESC[np.minimum(dcand, len(UNIT_DESC) - 1)])
    tgt_pos = dec_pos - 1                           # predicted from the previous token
    # MTP: predict decision j+1 from the token before decision j (within the same step)
    step_of = np.repeat(np.arange(T), L)
    same = np.zeros(nd, bool)
    same[:-1] = step_of[1:] == step_of[:-1]
    mtp_src = tgt_pos[:-1][same[:-1]] if nd > 1 else np.zeros(0, np.int64)
    mtp_idx = np.nonzero(same)[0] + 1
    return dict(S=S, T=T, ids=ids, otype=otype, dec_pos=dec_pos, dec_slot=dslot, dec_desc=desc, tgt_pos=tgt_pos,
                tgt_kind=kind, tgt_cand=dcand, val_pos=act_pos, mtp_pos=mtp_src, mtp_kind=kind[mtp_idx],
                mtp_cand=dcand[mtp_idx])


def collate(trajs, device="cpu", max_steps=None, value_every=4):
    lays = [layout(t, max_steps) for t in trajs]
    B, S = len(trajs), max(l["S"] for l in lays)
    ids = np.full((B, S), 8, np.int64)
    otype = np.full((B, S), -1, np.int8)
    cat = {k: [] for k in ("dec_pos", "dec_slot", "dec_desc", "tgt_pos", "tgt_kind", "tgt_cand", "tgt_w", "val_pos",
                           "val_tgt", "mtp_pos", "mtp_kind", "mtp_cand")}
    prod, glob, tiles, units = [], [], [], []
    for b, (tr, l) in enumerate(zip(trajs, lays)):
        ids[b, :l["S"]] = l["ids"]
        otype[b, :l["S"]] = l["otype"]
        w = (1.0 if tr["win"] > 0.5 else 0.5) * min(1.5, max(0.3, (tr["score"] - 1500) / 1500))
        bb = lambda x: np.stack([np.full(len(x), b), x], 1)
        cat["dec_pos"].append(bb(l["dec_pos"])); cat["dec_slot"].append(l["dec_slot"]); cat["dec_desc"].append(l["dec_desc"])
        cat["tgt_pos"].append(bb(l["tgt_pos"])); cat["tgt_kind"].append(l["tgt_kind"]); cat["tgt_cand"].append(l["tgt_cand"])
        cat["tgt_w"].append(np.full(len(l["tgt_pos"]), w, np.float32))
        v = l["val_pos"][::value_every]
        cat["val_pos"].append(bb(v)); cat["val_tgt"].append(np.tile([[tr["win"], tr["diff"]]], (len(v), 1)).astype(np.float32))
        cat["mtp_pos"].append(bb(l["mtp_pos"])); cat["mtp_kind"].append(l["mtp_kind"]); cat["mtp_cand"].append(l["mtp_cand"])
        T = l["T"]
        prod.append(tr["prod"][:T].reshape(-1, tr["prod"].shape[-1])); glob.append(tr["glob"][:T])
        tiles.append(tr["tiles"][:T]); units.append(tr["units"][:T].reshape(-1, tr["units"].shape[-1]))
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    out = {k: t(np.concatenate(v)) for k, v in cat.items()}
    out.update(ids=t(ids), otype=t(otype), prod=t(np.concatenate(prod)), glob=t(np.concatenate(glob)),
               tiles=t(np.concatenate(tiles)), units=t(np.concatenate(units)))
    return out
