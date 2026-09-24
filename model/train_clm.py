"""GPU pre-training / mid-training of the CLM-head decoder-only policy on ladder replays (distillation).

Objective (following Contrastive-LM/CLM train/finetune.py):
  * forward "choice" cross-entropy of every decision over its closed candidate set
    (logits = exp(logit_scale) * cos(state_head(h), action_head(candidate)))
  * backward in-batch InfoNCE: each distinct chosen candidate against all decision states of the batch,
    target = the states that chose it (normalised) -- CLM's bidirectional objective, averaged with forward
  * + value head (final win / money diff) + MTP (next-next decision) + DeepSeek-V3.2-style indexer KL
  * OneCycle LR (10% warm-up, cosine), AdamW, grad clip 1.0 (CLM defaults)

GPU-utilisation design
  * CPU work (npz decompression, index arithmetic) runs in background threads that keep a queue of
    ready batches in pinned memory; host->device copies are non_blocking.
  * Tile expansion to quadrant features happens on the GPU; AMP (bf16 if supported, else fp16+GradScaler).
  * Random step-crops of each trajectory (--crop_steps) give many distinct long sequences per epoch.

python -m model.train_seq --data "<glob of seq_*.npz>" --out ckpt_dir --stage pre
python -m model.train_seq --data ... --stage mid --init ckpt_dir/pre.pt --min_score 2800
"""
import argparse
import glob
import json
import math
import os
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.obs_tokens import OBS_TYPES
from data.seq_extract import load_trajs
from model.clm_batch import collate
from model.clm_policy import CLMPolicy
from model.dsv41 import ModelArgs
from model.seq_batch import obs_feats


def crop(tr, crop_steps, rng):
    T = len(tr["prod"])
    if not crop_steps or T <= crop_steps:
        return tr
    s0 = rng.randrange(0, T - crop_steps + 1)
    s1 = s0 + crop_steps
    off = tr["act_off"]
    a0, a1 = off[s0], off[s1]
    out = {k: tr[k][s0:s1] for k in ("prod", "glob", "tiles", "units")}
    out["act"] = tr["act"][a0:a1]
    out["act_off"] = off[s0:s1 + 1] - a0
    for k in ("win", "diff", "score"):
        out[k] = tr[k]
    return out


class Loader:
    """Background producer: files -> trajectories (shuffle buffer) -> collated pinned batches."""

    def __init__(self, files, batch, min_score, crop_steps, device, epochs=10 ** 9, seed=0, n_threads=3, qsize=6):
        self.files, self.batch, self.min_score = list(files), batch, min_score
        self.crop_steps, self.device, self.epochs = crop_steps, device, epochs
        self.q = queue.Queue(maxsize=qsize)
        self.traj_q = queue.Queue(maxsize=64)
        self.stop = False
        self.rng = random.Random(seed)
        threading.Thread(target=self._files, daemon=True).start()
        for i in range(n_threads):
            threading.Thread(target=self._batches, args=(seed + i + 1,), daemon=True).start()

    def _files(self):
        for ep in range(self.epochs):
            fs = self.files[:]
            self.rng.shuffle(fs)
            for f in fs:
                try:
                    trs = [t for t in load_trajs(f) if t["score"] >= self.min_score]
                except Exception as e:
                    print("load failed", f, e, flush=True)
                    continue
                self.rng.shuffle(trs)
                for t in trs:
                    self.traj_q.put(t)
        self.traj_q.put(None)

    def _batches(self, seed):
        rng = random.Random(seed)
        buf = []
        while not self.stop:
            t = self.traj_q.get()
            if t is None:
                self.traj_q.put(None)
                self.q.put(None)
                return
            buf.append(crop(t, self.crop_steps, rng))
            if len(buf) >= self.batch:
                b = collate(buf[:self.batch], device="cpu")
                if self.device.startswith("cuda"):
                    b = {k: v.pin_memory() for k, v in b.items()}
                self.q.put(b)
                buf = buf[self.batch:]

    def __iter__(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            yield {k: v.to(self.device, non_blocking=True) for k, v in b.items()}


def gpu_util():
    try:
        return torch.cuda.utilization()
    except Exception:
        return -1


def _bwd_infonce(lg, cand):
    """CLM backward direction: candidates -> states. lg [N, C] logits over the full candidate set,
    cand [N] chosen candidate per state. For each distinct chosen candidate, softmax over the batch's states,
    target = uniform over the states that chose it."""
    pool, inv = torch.unique(cand, return_inverse=True)
    sub = lg[:, pool].t()                                       # [P, N]
    tgt = torch.zeros_like(sub)
    tgt[inv, torch.arange(len(cand), device=lg.device)] = 1.0
    tgt = tgt / tgt.sum(1, keepdim=True)
    return -(tgt * F.log_softmax(sub.float(), 1)).sum(1).mean()


def compute_loss(net, b, dtype, coefs):
    feats = obs_feats(b, dtype)
    fwd = (lambda *x: net(*x)) if hasattr(net, "module") else net.forward_train
    lu, lm, value, mtp, aux = fwd(b["ids"], feats, b["otype"], b["dec_pos"], b["dec_slot"], b["dec_desc"],
                                  b["tgt_pos"], b["tgt_kind"], b["val_pos"], b["mtp_pos"], b["mtp_kind"])
    k, c, w = b["tgt_kind"], b["tgt_cand"], b["tgt_w"]
    cu, cm, wu, wm = c[k == 0], c[k == 1], w[k == 0], w[k == 1]
    ce_u = F.cross_entropy(lu.float(), cu, reduction="none")
    ce_m = F.cross_entropy(lm.float(), cm, reduction="none")
    ce = (torch.cat([ce_u * wu, ce_m * wm]).sum()) / w.sum()
    bwd = 0.5 * (_bwd_infonce(lu, cu) + _bwd_infonce(lm, cm)) if coefs["bwd"] > 0 else torch.zeros((), device=ce.device)
    vw = F.binary_cross_entropy_with_logits(value[:, 0].float(), b["val_tgt"][:, 0])
    vd = F.mse_loss(value[:, 1].float(), b["val_tgt"][:, 1])
    mtp_l = torch.zeros((), device=ce.device)
    if mtp is not None:
        mk, mc = b["mtp_kind"], b["mtp_cand"]
        mtp_l = 0.5 * (F.cross_entropy(mtp[0].float(), mc[mk == 0]) + F.cross_entropy(mtp[1].float(), mc[mk == 1]))
    # CLM averages forward and backward directions
    main = (ce + coefs["bwd"] * bwd) / (1 + coefs["bwd"])
    loss = main + coefs["value"] * (vw + 0.1 * vd) + coefs["mtp"] * mtp_l + coefs["index"] * aux
    with torch.no_grad():
        acc_u = (lu.argmax(-1) == cu).float().mean() if len(cu) else torch.zeros(())
        acc_m = (lm.argmax(-1) == cm).float().mean() if len(cm) else torch.zeros(())
    core = net.module if hasattr(net, "module") else net
    return loss, dict(ce=float(ce), bwd=float(bwd), acc_unit=float(acc_u), acc_market=float(acc_m), vwin=float(vw),
                      vdiff=float(vd), mtp=float(mtp_l), idx=float(aux), scale=float(core.scale()),
                      ntok=int(c.numel()), seq=int(b["ids"].numel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob(s) of seq_*.npz, comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["pre", "mid"], default="pre")
    ap.add_argument("--init", default=None)
    ap.add_argument("--min_score", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--crop_steps", type=int, default=240)
    ap.add_argument("--val_files", type=int, default=2)
    ap.add_argument("--max_hours", type=float, default=11.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--bwd", type=float, default=1.0, help="weight of CLM backward in-batch InfoNCE")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    # DDP (torchrun) support: one process per GPU, each reading a disjoint subset of files
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    ms = a.min_score if a.min_score is not None else (1800 if a.stage == "pre" else 2800)
    lr = a.lr or (1e-3 if a.stage == "pre" else 2e-4)
    files = sorted(sum([glob.glob(g) for g in a.data.split(",")], []))
    random.Random(0).shuffle(files)
    nv = a.val_files if len(files) > a.val_files else 0
    val, train = files[:nv], files[nv:]
    train = train[rank::world] if world > 1 else train
    os.makedirs(a.out, exist_ok=True)
    args = ModelArgs(obs_types=OBS_TYPES, dim=a.dim, n_layers=a.n_layers,
                     compress_ratios=tuple([0] + [2] * ((a.n_layers - 2) // 2) + [1] * (a.n_layers - 2 - (a.n_layers - 2) // 2) + [0, 0]),
                     kv_source_layers=(1, 1 + (a.n_layers - 2) // 2), index_source_layers=(1, 1 + (a.n_layers - 2) // 2))
    json.dump(args.__dict__, open(os.path.join(a.out, "model_args.json"), "w"), default=list)
    net = CLMPolicy(args).to(device)
    if a.init:
        sd = torch.load(a.init, map_location=device)
        print("init", net.load_state_dict(sd, strict=False), flush=True)
    core = net
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                                        find_unused_parameters=True)
        core = net.module
    n_params = sum(p.numel() for p in net.parameters())
    print(f"device={device} params={n_params} train_files={len(train)} val_files={len(val)} min_score={ms}", flush=True)
    use_amp = device.startswith("cuda")
    # bf16 only with native support (Ampere+); T4 (sm75) would emulate it slowly -> fp16 + GradScaler
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    print("amp dtype", amp_dtype, "world", world, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    coefs = {"value": 0.2, "mtp": 0.3, "index": 0.1, "bwd": a.bwd}
    loader = Loader(train, a.batch, ms, a.crop_steps, device, epochs=a.epochs)
    print("loader started", flush=True)
    deadline = time.time() + a.max_hours * 3600
    t0, tl = time.time(), time.time()
    step, tokens, seq_tokens = 0, 0, 0
    warm = 200
    est_total = max(1000, int(a.epochs * len(train) * 400 / a.batch))   # rough: ~400 trajs per file
    net.train()
    for i, b in enumerate(loader):
        if i == 0:
            print("first batch", {k: tuple(v.shape) for k, v in b.items()}, flush=True)
        prog = min(1.0, step / est_total)
        cur_lr = lr * (prog / 0.1 if prog < 0.1 else 0.5 * (1 + math.cos(math.pi * (prog - 0.1) / 0.9)) * 0.96 + 0.04)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss, st = compute_loss(net, b, amp_dtype if use_amp else torch.float32, coefs)
        scaler.scale(loss / a.accum).backward()
        if (i + 1) % a.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            core.update_gate_bias()
            step += 1
        tokens += st["ntok"]
        seq_tokens += st["seq"]
        if (step % a.log_every == 0 or step <= 3) and (i + 1) % a.accum == 0:
            dt = time.time() - tl
            print(json.dumps({"step": step, "lr": round(cur_lr, 6), **{k: round(v, 4) if isinstance(v, float) else v
                                                                          for k, v in st.items()},
                              "seq_tok_per_s": round(seq_tokens / max(dt, 1e-6)), "gpu_util": gpu_util(),
                              "queue": loader.q.qsize(), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                              if use_amp else 0, "rank": rank, "elapsed_min": round((time.time() - t0) / 60, 1)}), flush=True)
            tl, seq_tokens = time.time(), 0
        if step and step % a.save_every == 0 and (i + 1) % a.accum == 0 and rank == 0:
            torch.save(core.state_dict(), os.path.join(a.out, f"{a.stage}.pt"))
        if time.time() > deadline:
            break
    loader.stop = True
    if rank == 0:
        torch.save(core.state_dict(), os.path.join(a.out, f"{a.stage}.pt"))
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        if rank != 0:
            return
    net = core
    # validation
    if not val:
        print("saved", os.path.join(a.out, f"{a.stage}.pt"), flush=True)
        return
    net.eval()
    vl = Loader(val, a.batch, ms, a.crop_steps, device, epochs=1, seed=123)
    vs = []
    with torch.no_grad():
        for j, b in enumerate(vl):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vs.append(compute_loss(net, b, amp_dtype if use_amp else torch.float32, coefs)[1])
            if j >= 30:
                break
    if vs:
        print("VAL", json.dumps({k: round(float(np.mean([v[k] for v in vs])), 4) for k in vs[0]}), flush=True)
    print("saved", os.path.join(a.out, f"{a.stage}.pt"), flush=True)


if __name__ == "__main__":
    main()
