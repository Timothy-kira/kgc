"""Kaggle GPU kernel: pooled offline learner (rl/v5_offline.py) for the A track, two variants.

python notebooks/build_v5offline_kernel.py <kernel_dir> <owner> <data_dataset> [hours_per_variant=1.5]
Code from <owner>/kgc-src (or V5_DATASET); <data_dataset> holds samples/**/samples_*.pkl (all sampler kernels),
engram.pt (new Engram tables) and opp_vocab.npz. Variants: fresh Engram tables and --eng_init engram.pt
--freeze_tables. Output: /kaggle/working/{fresh,eng}/policy.{pt,json}, grid.json, best.txt (variant with the higher
held-out decision value).
"""
import json
import os
import sys

SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
src = glob.glob("/kaggle/input/**/rl/v5_offline.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
env = dict(os.environ, PYTHONPATH="/kaggle/working/src")
pk = glob.glob("/kaggle/input/**/samples_*.pkl", recursive=True)
eng = [p for p in glob.glob("/kaggle/input/**/engram.pt", recursive=True)]
voc = [p for p in glob.glob("/kaggle/input/**/opp_vocab.npz", recursive=True)]
print("samples files", len(pk), "engram", eng, "vocab", voc, flush=True)
subprocess.run(["nvidia-smi"], check=False)
res = {}
for name, extra in (("eng", ["--eng_init", eng[0], "--freeze_tables"] if eng else None), ("fresh", [])):
    if extra is None:
        continue
    out = "/kaggle/working/" + name
    cmd = [sys.executable, "-u", "-m", "rl.v5_offline", "--samples", "/kaggle/input/**/samples_*.pkl", "--vocab", voc[0],
           "--out", out, "--hours", "__HOURS__", "--epochs", "40", "--bs", "1024"] + extra + __EXTRA__
    print("cmd", cmd, flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
    print(name, "exit", p.wait(), flush=True)
    if os.path.exists(out + "/policy.json"):
        res[name] = json.load(open(out + "/policy.json"))["hold_gain"]
print("RESULT", json.dumps(res), flush=True)
if res:
    open("/kaggle/working/best.txt", "w").write(max(res, key=res.get))
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
'''

if __name__ == "__main__":
    kdir, owner, data_ds = sys.argv[1:4]
    hours = sys.argv[4] if len(sys.argv) > 4 else "1.5"
    os.makedirs(kdir, exist_ok=True)
    extra = os.environ.get("V5OFF_EXTRA", "").split()
    open(os.path.join(kdir, "script.py"), "w").write(SCRIPT.replace("__HOURS__", hours).replace("__EXTRA__", repr(extra)))
    json.dump({"id": f"{owner}/kgc-v5offline", "title": "kgc v5offline", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{owner}/" + os.environ.get("V5_DATASET", "kgc-src"), data_ds],
               "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
