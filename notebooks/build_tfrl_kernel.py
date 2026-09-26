"""Kaggle GPU kernel for the B track (rl/tf_rl.py, docs/PLAN_RL.md): pure-Transformer policy RL.

python notebooks/build_tfrl_kernel.py <kernel_dir> <owner> [hours=5.6] [extra args for rl.tf_rl]
Code, assets (rl_pools.pkl, opp_vocab.npz) and the Engram checkpoint (ckpt/engram.pt + model_args.json) come from
the private dataset <owner>/kgc-src2. Slug kgc-gpu-tfrl (tools/kaggle_gpu.py keeps one GPU notebook alive).
Output: /kaggle/working/tf/{log.jsonl, tf_latest.pt, tf_best.pt, tf_phase1.pt, best.json, base_eval.json}.
"""
import json
import os
import sys

SCRIPT = r'''
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
src = glob.glob("/kaggle/input/**/rl/tf_rl.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt"))
ck = os.path.dirname(glob.glob("/kaggle/input/**/ckpt/engram.pt", recursive=True)[0])
os.chdir("/kaggle/working/src")
subprocess.run(["nvidia-smi"], check=False)
env = dict(os.environ, PYTHONPATH="/kaggle/working/src")
cmd = [sys.executable, "-u", "-m", "rl.tf_rl", "--ckpt", ck, "--pools", "assets/rl_pools.pkl",
       "--vocab", "assets/opp_vocab.npz", "--out", "/kaggle/working/tf", "--hours", "__HOURS__"] + __EXTRA__
print("cmd", cmd, flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
for line in p.stdout:
    print(line, end="", flush=True)
print("exit", p.wait(), flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
print("kept", sorted(os.listdir("/kaggle/working/tf")), flush=True)
'''

if __name__ == "__main__":
    kdir, owner = sys.argv[1:3]
    hours = sys.argv[3] if len(sys.argv) > 3 else "5.6"
    extra = sys.argv[4:]
    os.makedirs(kdir, exist_ok=True)
    open(os.path.join(kdir, "script.py"), "w").write(SCRIPT.replace("__HOURS__", hours).replace("__EXTRA__", repr(extra)))
    json.dump({"id": f"{owner}/kgc-gpu-tfrl", "title": "kgc gpu tfrl", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{owner}/kgc-src2"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
