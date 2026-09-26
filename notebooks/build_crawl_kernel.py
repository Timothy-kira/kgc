"""Kaggle CPU kernel: one partition of the replay crawler (data/crawl.py --part k/n), resuming from the public DB.

python notebooks/build_crawl_kernel.py <kernel_dir> <owner> <k> <n> [hours=11.3]
Mounts xishengfeng/kaggriculture-replay-db (public). Output /kaggle/working/crawl_out: the new shards
(ep_*_p<k>.parquet), state.json and index/ - not the copied DB, so the download stays small. Merge = copy the new
shards of every partition into the local replay_db/shards.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["data/__init__.py", "data/crawl.py", "data/replay_db.py", "env/__init__.py", "env/fast_env.py",
         "env/replay_check.py"]

SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys
for pkg in ("zstandard", "pyarrow"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
SRC = "/kaggle/working/src"
for rel, text in __FILES__.items():
    p = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)
sys.path.insert(0, SRC)
best, best_n = None, -1
for st in glob.glob("/kaggle/input/**/state.json", recursive=True):
    if not os.path.isdir(os.path.join(os.path.dirname(st), "index")):
        continue
    n = json.load(open(st)).get("n_downloaded", 0)
    if n > best_n:
        best, best_n = os.path.dirname(st), n
print("resume source:", best, best_n, flush=True)
if best is None:
    subprocess.run("find /kaggle/input -maxdepth 4 | head -50", shell=True)
    sys.exit("no resume DB under /kaggle/input")
OUT = "/kaggle/working/replay_db"
from data import crawl
crawl.main(["--out", OUT, "--resume-from", best, "--min-score", "1500", "--max-hours", "__HOURS__", "--rounds", "1000",
            "--max-lists", "300", "--workers", "12", "--list-workers", "2", "--shard-size", "250",
            "--part", "__K__/__N__"])
out = "/kaggle/working/crawl_out"
os.makedirs(os.path.join(out, "shards"), exist_ok=True)
new = glob.glob(os.path.join(OUT, "shards", "ep_*_p__K__.parquet"))
for f in new:
    shutil.move(f, os.path.join(out, "shards", os.path.basename(f)))
shutil.copy(os.path.join(OUT, "state.json"), out)
shutil.copytree(os.path.join(OUT, "index"), os.path.join(out, "index"), dirs_exist_ok=True)
shutil.rmtree(OUT, ignore_errors=True)
shutil.rmtree(SRC, ignore_errors=True)
print("new shards:", len(new), flush=True)
'''

if __name__ == "__main__":
    kdir, owner, k, n = sys.argv[1:5]
    hours = sys.argv[5] if len(sys.argv) > 5 else "11.3"
    os.makedirs(kdir, exist_ok=True)
    files = {rel: open(os.path.join(ROOT, rel)).read() for rel in FILES}
    code = SCRIPT.replace("__FILES__", repr(files)).replace("__HOURS__", hours).replace("__K__", k).replace("__N__", n)
    open(os.path.join(kdir, "script.py"), "w").write(code)
    json.dump({"id": f"{owner}/kgc-crawl-{k}", "title": f"kgc crawl {k}", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": ["xishengfeng/kaggriculture-replay-db"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
