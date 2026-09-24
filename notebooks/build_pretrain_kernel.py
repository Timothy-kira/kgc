"""Build a Kaggle kernel (GPU account) that extracts replay features and pretrains the trunk.

Inputs:  public replay DB dataset + a code dataset with this repo's sources.
Output:  /kaggle/working/pretrained.pt / pretrained.npz (+ extracted parts for reuse)

python notebooks/build_pretrain_kernel.py <kernel_dir> <owner> <code_dataset_slug> [min_score] [max_eps] [epochs]
"""
import json
import os
import sys

TEMPLATE = r'''
import glob, os, subprocess, sys, shutil
for pkg in ("zstandard", "pyarrow", "kaggle_environments"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg.replace("_", "-")], check=False)
code = glob.glob("/kaggle/input/**/kgc_src/agent/features.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(code))
shutil.copytree(root, "/kaggle/working/src", dirs_exist_ok=True)
sys.path.insert(0, "/kaggle/working/src")
os.chdir("/kaggle/working/src")
dbs = sorted(glob.glob("/kaggle/input/**/replay_db/state.json", recursive=True))
import json as _j
db = max(dbs, key=lambda p: _j.load(open(p)).get("n_downloaded", 0))
db = os.path.dirname(db)
print("DB", db, flush=True)
ncpu = os.cpu_count()
subprocess.run([sys.executable, "-m", "data.extract", db, "/kaggle/working/ext", "{min_score}", "{max_eps}", str(ncpu)], check=True)
import torch
print("cuda", torch.cuda.is_available(), flush=True)
subprocess.run([sys.executable, "-m", "model.pretrain", "/kaggle/working/ext", "/kaggle/working/pretrained.pt", "{epochs}"], check=True)
'''


def build(kdir, owner, code_ds, min_score="2600", max_eps="6000", epochs="4"):
    os.makedirs(kdir, exist_ok=True)
    code = TEMPLATE.replace("{min_score}", min_score).replace("{max_eps}", max_eps).replace("{epochs}", epochs)
    open(os.path.join(kdir, "pretrain.py"), "w").write(code)
    meta = {"id": f"{owner}/kgc-pretrain", "title": "kgc pretrain", "code_file": "pretrain.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [code_ds, "xishengfeng/kaggriculture-replay-db"],
            "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)


if __name__ == "__main__":
    build(*sys.argv[1:])
