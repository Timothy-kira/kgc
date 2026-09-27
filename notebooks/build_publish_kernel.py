"""Kaggle CPU kernel: republish other kernels' outputs as a dataset, so another account can mount them.

KAGGLE_API_TOKEN=<owner's token> python notebooks/build_publish_kernel.py <kernel_dir> <owner> <dataset_slug> <kernel,kernel,...>
     [patterns=seq_*.npz,ep_*.npz] [public=1]
The kernel mounts the given kernels' outputs, symlinks every matching file (flat, names must be unique) into one
upload folder and runs `kaggle datasets create` (or `version` if it exists). The token is injected into the generated
script only (never into the repo); delete the kernel after it finishes.
"""
import json
import os
import sys

SCRIPT = r'''
import glob, json, os, subprocess, sys
os.environ["KAGGLE_API_TOKEN"] = __TOKEN__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=False)
up = "/kaggle/working/up"
os.makedirs(up, exist_ok=True)
n, size = 0, 0
for pat in __PATTERNS__:
    for f in glob.glob("/kaggle/input/**/" + pat, recursive=True):
        dst = os.path.join(up, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
            n += 1
            size += os.path.getsize(f)
print("files", n, "GB", round(size / 1e9, 2), flush=True)
json.dump({"title": __SLUG__.split("/")[1], "id": __SLUG__, "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(up, "dataset-metadata.json"), "w"))
r = subprocess.run(["kaggle", "datasets", "status", __SLUG__], capture_output=True, text=True)
exists = r.returncode == 0 and "ready" in r.stdout
cmd = (["kaggle", "datasets", "version", "-p", up, "-m", "republish", "-q", "-r", "skip"] if exists else
       ["kaggle", "datasets", "create", "-p", up, "-q", "-r", "skip"] + (["--public"] if __PUBLIC__ else []))
print("cmd", cmd, flush=True)
rc = subprocess.run(cmd).returncode
print("upload exit", rc, flush=True)
subprocess.run(["kaggle", "datasets", "status", __SLUG__])
'''

if __name__ == "__main__":
    kdir, owner, slug, kernels = sys.argv[1:5]
    patterns = (sys.argv[5] if len(sys.argv) > 5 else "seq_*.npz,ep_*.npz").split(",")
    public = (sys.argv[6] if len(sys.argv) > 6 else "1") == "1"
    token = os.environ["KAGGLE_API_TOKEN"]
    os.makedirs(kdir, exist_ok=True)
    code = (SCRIPT.replace("__TOKEN__", repr(token)).replace("__PATTERNS__", repr(patterns))
            .replace("__SLUG__", repr(slug)).replace("__PUBLIC__", repr(public)))
    open(os.path.join(kdir, "script.py"), "w").write(code)
    json.dump({"id": f"{owner}/kgc-publish", "title": "kgc publish", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": kernels.split(",")},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
