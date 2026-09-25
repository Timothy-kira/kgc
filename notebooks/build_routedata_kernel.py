"""Kaggle CPU kernel: H5 route-value data (tools/route_data.py) for a range of jobs.

python notebooks/build_routedata_kernel.py <kernel_dir> <owner> <idx> <job_from> <job_to> [hours=10.5]
Code comes from the private dataset <owner>/kgc-src (agent, data, env, model, infra, tools, league), the replay
DB from xishengfeng/kaggriculture-replay-db. Output: /kaggle/working/route_<idx>.jsonl
"""
import json
import os
import sys

SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys
for pkg in ("zstandard", "pyarrow"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
# the replay-verified engine version (env/replay_check.py: bit-identical to downloaded ladder replays)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
src = [p for p in glob.glob("/kaggle/input/**/agent/opp_events.py", recursive=True)][0]
root = os.path.dirname(os.path.dirname(src))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
sys.path.insert(0, "/kaggle/working/src")
dbs = glob.glob("/kaggle/input/**/replay_db/state.json", recursive=True)
db = os.path.dirname(max(dbs, key=lambda p: json.load(open(p)).get("n_downloaded", 0)))
env = dict(os.environ, ROUTE_MAX_HOURS="__HOURS__", PYTHONPATH="/kaggle/working/src")
subprocess.run([sys.executable, "-m", "tools.route_data", "/kaggle/working/route___IDX__.jsonl", "__A__", "__B__",
                str(os.cpu_count()), db], check=False, env=env)
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
print("lines", sum(1 for _ in open("/kaggle/working/route___IDX__.jsonl")))
'''

if __name__ == "__main__":
    kdir, owner, idx, a, b = sys.argv[1:6]
    hours = sys.argv[6] if len(sys.argv) > 6 else "10.5"
    os.makedirs(kdir, exist_ok=True)
    open(os.path.join(kdir, "script.py"), "w").write(
        SCRIPT.replace("__IDX__", idx).replace("__A__", a).replace("__B__", b).replace("__HOURS__", hours))
    json.dump({"id": f"{owner}/kgc-routedata-{idx}", "title": f"kgc routedata {idx}", "code_file": "script.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
               "enable_internet": True, "dataset_sources": [f"{owner}/kgc-src", "xishengfeng/kaggriculture-replay-db"],
               "competition_sources": [], "kernel_sources": []}, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
