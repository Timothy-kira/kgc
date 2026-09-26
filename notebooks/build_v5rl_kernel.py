"""Kaggle CPU kernel for the A track (rl/v5_rl.py, docs/PLAN_RL.md).

python notebooks/build_v5rl_kernel.py <kernel_dir> <owner> [hours=11.2] [extra args for rl.v5_rl]
Code + assets (rl_pools.pkl, opp_vocab.npz) come from the private dataset <owner>/kgc-src. Output:
/kaggle/working/v5/{log.jsonl, policy_latest.pt, policy_evalNNN.pt, best.json, base_eval.json}.
"""
import json
import os
import sys

SCRIPT = r'''
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
src = glob.glob("/kaggle/input/**/rl/v5_rl.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
env = dict(os.environ, PYTHONPATH="/kaggle/working/src")
out = "/kaggle/working/v5"
cmd = [sys.executable, "-u", "-m", "rl.v5_rl", "--pools", "assets/rl_pools.pkl", "--vocab", "assets/opp_vocab.npz",
       "--out", out, "--hours", "__HOURS__", "--procs", str(os.cpu_count())] + __EXTRA__
print("cmd", cmd, flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
for line in p.stdout:
    print(line, end="", flush=True)
print("exit", p.wait(), flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
evals = sorted(glob.glob(out + "/policy_eval*.pt"))
best = None
if os.path.exists(out + "/best.json"):
    import json
    best = json.load(open(out + "/best.json")).get("best")
for f in evals[:-3]:
    if os.path.basename(f) != best:
        os.remove(f)
print("kept", sorted(os.listdir(out)), flush=True)
'''

if __name__ == "__main__":
    kdir, owner = sys.argv[1:3]
    hours = sys.argv[3] if len(sys.argv) > 3 else "11.2"
    extra = sys.argv[4:]
    suffix = os.environ.get("V5_SUFFIX", "")             # parallel replicas: kgc-v5rl-b, kgc-v5rl-c, ...
    os.makedirs(kdir, exist_ok=True)
    open(os.path.join(kdir, "script.py"), "w").write(SCRIPT.replace("__HOURS__", hours).replace("__EXTRA__", repr(extra)))
    json.dump({"id": f"{owner}/kgc-v5rl{suffix}", "title": f"kgc v5rl{suffix}", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{owner}/" + os.environ.get("V5_DATASET", "kgc-src")], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    if os.environ.get("V5_ACCEL") == "tpu":                  # TPU VM host = 224 CPU cores for the fork rollouts
        m = json.load(open(os.path.join(kdir, "kernel-metadata.json")))
        m.update(enable_tpu=True, machine_shape="Tpu1VmV38")
        json.dump(m, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
