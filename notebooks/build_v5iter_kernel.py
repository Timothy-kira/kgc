"""RTX PRO 6000 kernel (via the ARC Prize competition: internet OFF) for rl/v5_iter.py - large-scale iterated RL.

python notebooks/build_v5iter_kernel.py <kernel_dir> <owner> [hours=11.3] [extra v5_iter args...]
Inputs (mounted, no network): <owner>/kgc-src (code + assets + assets/cand/*.pt), <owner>/kgc-wheels (offline wheels:
kaggle-environments 1.32.7 --no-deps, zstandard), sample datasets / kernels given by V5I_DATA (datasets) and
V5I_KERNELS (kernel outputs, e.g. the Engram trainer), comma separated. Push with
`kaggle kernels push -p <dir> --accelerator NvidiaRtxPro6000`.
Output: /kaggle/working/iter/ (round_k/{learn/policy.pt, recheck.json}, best.json).
"""
import json
import os
import sys

SCRIPT = r'''
import glob, os, shutil, subprocess, sys
whl = glob.glob("/kaggle/input/**/kaggle_environments-*.whl", recursive=True) + glob.glob("/kaggle/input/**/zstandard-*.whl", recursive=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"] + whl, check=False)
src = glob.glob("/kaggle/input/**/rl/v5_iter.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
os.environ["PYTHONPATH"] = "/kaggle/working/src"
subprocess.run(["nvidia-smi"], check=False)
print("cpus", os.cpu_count(), flush=True)
def first(pat):
    f = sorted(glob.glob(pat, recursive=True))
    return f[0] if f else ""
eng = first("/kaggle/input/**/ckpt/engram.pt") or first("/kaggle/input/**/engram.pt")
seed = ",".join(sorted(set(os.path.dirname(p) + "/samples_*.pkl" for p in glob.glob("/kaggle/input/**/samples_*.pkl", recursive=True))))
print("engram", eng, "seed dirs", seed.count(",") + (1 if seed else 0), flush=True)
cmd = [sys.executable, "-u", "-m", "rl.v5_iter", "--out", "/kaggle/working/iter", "--pools", "assets/rl_pools.pkl",
       "--eval_pools", "assets/rl_pools_v2.pkl", "--vocab", "assets/opp_vocab.npz", "--seed_samples", seed,
       "--engram", eng, "--hours", "__HOURS__"] + __EXTRA__
print("cmd", cmd, flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
print("exit", p.wait(), flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
for d in glob.glob("/kaggle/working/iter/round_*/samples"):          # keep outputs small: samples stay in the kernel
    shutil.rmtree(d, ignore_errors=True)
'''

if __name__ == "__main__":
    kdir, owner = sys.argv[1:3]
    hours = sys.argv[3] if len(sys.argv) > 3 else "11.3"
    extra = sys.argv[4:]
    os.makedirs(kdir, exist_ok=True)
    open(os.path.join(kdir, "script.py"), "w").write(SCRIPT.replace("__HOURS__", hours).replace("__EXTRA__", repr(extra)))
    data = [d for d in os.environ.get("V5I_DATA", "").split(",") if d]
    kern = [k for k in os.environ.get("V5I_KERNELS", "").split(",") if k]
    json.dump({"id": f"{owner}/kgc-v5iter", "title": "kgc v5iter", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
               "dataset_sources": [f"{owner}/kgc-src", f"{owner}/kgc-wheels"] + data,
               "competition_sources": ["arc-prize-2026-arc-agi-3"], "kernel_sources": kern},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
