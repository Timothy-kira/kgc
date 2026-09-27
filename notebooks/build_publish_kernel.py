"""Kaggle CPU kernel: republish other kernels' outputs as a dataset, so another account can mount them.

KAGGLE_API_TOKEN=<owner's token> python notebooks/build_publish_kernel.py <kernel_dir> <owner> <dataset_slug> <kernel,kernel,...>
     [patterns=seq_*.npz,ep_*.npz] [public=1]
The kernel mounts the given kernels' outputs and splits every matching file (flat, names must be unique) over CHUNKS
(env, default 8) datasets <dataset_slug>-<i>, each uploaded as ONE zip archive of a data/ folder (Kaggle unpacks it),
strictly one after another with a timeout, stopping at the first failure (a CreateDataset call with ~1800 files times
out with a 504 and leaves the upload hanging). The token is injected into the generated
script only (never into the repo); delete the kernel after it finishes.
"""
import json
import os
import sys

SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys, time
os.environ["KAGGLE_API_TOKEN"] = __TOKEN__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=False)
files = {}
for pat in __PATTERNS__:
    for f in glob.glob("/kaggle/input/**/" + pat, recursive=True):
        files.setdefault(os.path.basename(f), f)
names = sorted(files)
k = __CHUNKS__
print("files", len(names), "GB", round(sum(os.path.getsize(files[n]) for n in names) / 1e9, 2), "chunks", k, flush=True)

def status(slug):
    r = subprocess.run(["kaggle", "datasets", "status", slug], capture_output=True, text=True, timeout=120)
    return (r.stdout + r.stderr).strip()

t0 = time.time()
for i in range(k):
    slug = f"{__SLUG__}-{i}"
    if status(slug) == "ready":
        print(slug, "exists", flush=True)
        continue
    up = f"/kaggle/working/up{i}"
    data = os.path.join(up, "data")                 # one directory -> one zip archive -> one file per CreateDataset
    os.makedirs(data, exist_ok=True)
    for n in names[i::k]:
        os.symlink(files[n], os.path.join(data, n))
    json.dump({"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(up, "dataset-metadata.json"), "w"))
    cmd = ["kaggle", "datasets", "create", "-p", up, "-q", "-r", "zip"] + (["--public"] if __PUBLIC__ else [])
    try:
        rc = subprocess.run(cmd, timeout=2400).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    print(slug, "create exit", rc, round((time.time() - t0) / 60, 1), "min", flush=True)
    st = ""
    for _ in range(30):                               # <= 15 min for the server to finish processing
        st = status(slug)
        if st == "ready":
            break
        time.sleep(30)
    print(slug, "status", st, flush=True)
    shutil.rmtree(up, ignore_errors=True)
    if st != "ready":                                 # fail fast: never leave uploads hanging
        print("STOP: chunk", i, "not ready", flush=True)
        sys.exit(1)
print("ALL READY", flush=True)
'''

if __name__ == "__main__":
    kdir, owner, slug, kernels = sys.argv[1:5]
    patterns = (sys.argv[5] if len(sys.argv) > 5 else "seq_*.npz,ep_*.npz").split(",")
    public = (sys.argv[6] if len(sys.argv) > 6 else "1") == "1"
    token = os.environ["KAGGLE_API_TOKEN"]
    os.makedirs(kdir, exist_ok=True)
    chunks = int(os.environ.get("CHUNKS", "8"))
    code = (SCRIPT.replace("__TOKEN__", repr(token)).replace("__PATTERNS__", repr(patterns))
            .replace("__SLUG__", repr(slug)).replace("__PUBLIC__", repr(public)).replace("__CHUNKS__", str(chunks)))
    open(os.path.join(kdir, "script.py"), "w").write(code)
    json.dump({"id": f"{owner}/kgc-publish", "title": "kgc publish", "code_file": "script.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": kernels.split(",")},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
