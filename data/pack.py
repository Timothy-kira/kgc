"""Pack extracted replay trajectories into large fixed-shape chunks for GPU training.

CPU-heavy work (re-simulation, feature extraction, decompression, padding) happens once here, so
the GPU trainer only memory-maps contiguous float16/int8 arrays and copies whole chunks to the
device (see model/pretrain.py ChunkLoader). Each chunk directory holds:
    P.npy   float16 [N, T, 9, PF]     G.npy float16 [N, T, GF]     y.npy int8 [N, T, 9]
    len.npy int16 [N]                 day_tgt.npy float16 [N, ND, N_TGT]  day_mask.npy int8 [N, ND]
    win.npy / diff.npy / score.npy float32 [N]
Packing is cheap (I/O bound), so it simply rebuilds all chunks from all extracted parts.

python -m data.pack <ext_dir> <pack_dir> [chunk_size]
"""
import glob
import os
import sys

import numpy as np

T = 720
ND = 30


def main():
    src, out = sys.argv[1], sys.argv[2]
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    parts = sorted(glob.glob(src + "/part_*.npz"))
    cid = 0
    buf = []

    def flush(items):
        nonlocal cid
        n = len(items)
        d = os.path.join(out, f"chunk_{cid:04d}.tmp")
        os.makedirs(d, exist_ok=True)
        pf, gf, nt = items[0]["P"].shape[-1], items[0]["G"].shape[-1], items[0]["day_tgt"].shape[-1]
        arr = {
            "P": np.zeros((n, T, 9, pf), np.float16), "G": np.zeros((n, T, gf), np.float16),
            "y": np.zeros((n, T, 9), np.int8), "len": np.zeros(n, np.int16),
            "day_tgt": np.zeros((n, ND, nt), np.float16), "day_mask": np.zeros((n, ND), np.int8),
            "win": np.zeros(n, np.float32), "diff": np.zeros(n, np.float32), "score": np.zeros(n, np.float32),
        }
        for i, it in enumerate(items):
            L = min(T, len(it["P"]))
            arr["P"][i, :L], arr["G"][i, :L], arr["y"][i, :L] = it["P"][:L], it["G"][:L], it["y"][:L]
            arr["len"][i] = L
            k = min(ND, len(it["day_mask"]))
            arr["day_tgt"][i, :k], arr["day_mask"][i, :k] = it["day_tgt"][:k], it["day_mask"][:k]
            arr["win"][i], arr["diff"][i], arr["score"][i] = it["win"], it["diff"], it["score"]
        for k, v in arr.items():
            np.save(os.path.join(d, k + ".npy"), v)
        os.replace(d, d[:-4])
        cid += 1
        print("chunk", d[:-4], n, flush=True)

    for p in parts:
        try:
            z = np.load(p)
            n = int(z["n"])
            for j in range(n):
                buf.append({k: z[f"{k}_{j}"] for k in ("P", "G", "y", "day_tgt", "day_mask", "win", "diff", "score")})
        except Exception as e:
            print("skip", p, e, flush=True)
            continue
        if len(buf) >= chunk:
            flush(buf[:chunk])
            buf = buf[chunk:]
    if buf:
        flush(buf)


if __name__ == "__main__":
    main()
