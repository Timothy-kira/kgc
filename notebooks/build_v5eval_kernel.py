"""Kaggle CPU kernel: A-track promotion re-check (tools/v5_recheck.py) of policies shipped in <owner>/kgc-src.

python notebooks/build_v5eval_kernel.py <kernel_dir> <owner> <policy path in dataset, e.g. assets/cand/policy_eval010.pt>
       [margins=0.25,0.5,1.0] [extra env, e.g. ENS=1]
Output: /kaggle/working/recheck.json (+ the log lines). Delete the kernel after reading the result.
"""
import json
import os
import sys

SCRIPT = r'''
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
src = glob.glob("/kaggle/input/**/tools/v5_recheck.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
env = dict(os.environ, PYTHONPATH="/kaggle/working/src")
cmd = [sys.executable, "-u", "-m", "tools.v5_recheck", "__POLICY__", "assets/rl_pools_v2.pkl", "assets/opp_vocab.npz",
       "/kaggle/working/recheck.json", "__MARGINS__", str(os.cpu_count())]
print("cmd", cmd, flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
for line in p.stdout:
    print(line, end="", flush=True)
print("exit", p.wait(), flush=True)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
'''

if __name__ == "__main__":
    kdir, owner, policy = sys.argv[1:4]
    margins = sys.argv[4] if len(sys.argv) > 4 else "0.25,0.5,1.0"
    os.makedirs(kdir, exist_ok=True)
    open(os.path.join(kdir, "script.py"), "w").write(SCRIPT.replace("__POLICY__", policy).replace("__MARGINS__", margins))
    json.dump({"id": f"{owner}/kgc-v5eval", "title": "kgc v5eval", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{owner}/kgc-src"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
