"""Helpers to build self-contained Kaggle kernels that embed this repo's sources.

The generated script writes the embedded files to /kaggle/working/src, installs missing packages,
and then runs `body` (python source) with that directory on sys.path.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = ["agent/__init__.py", "agent/action_tokens.py", "agent/features.py", "agent/obs_tokens.py",
                 "data/__init__.py", "data/crawl.py", "data/replay_db.py", "data/seq_extract.py",
                 "env/__init__.py", "env/fast_env.py", "env/replay_check.py",
                 "model/__init__.py", "model/dsv41.py", "model/seq_batch.py", "model/train_seq.py"]

HEADER = r'''
import os, subprocess, sys
for pkg in ("zstandard", "pyarrow", "kaggle_environments"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg.replace("_", "-")], check=False)
SRC = "/kaggle/working/src" if os.path.isdir("/kaggle/working") else os.path.abspath("src")
FILES = __FILES__
for rel, text in FILES.items():
    p = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)
sys.path.insert(0, SRC)
os.chdir(SRC)
'''


def build_kernel(kdir, slug, title, body, files=None, gpu=False, internet=True, private=True,
                 dataset_sources=(), kernel_sources=()):
    os.makedirs(kdir, exist_ok=True)
    files = files or DEFAULT_FILES
    src = {rel: open(os.path.join(ROOT, rel)).read() for rel in files if os.path.exists(os.path.join(ROOT, rel))}
    code = HEADER.replace("__FILES__", repr(src)) + "\n" + body
    open(os.path.join(kdir, "script.py"), "w").write(code)
    meta = {"id": slug, "title": title, "code_file": "script.py", "language": "python", "kernel_type": "script",
            "is_private": private, "enable_gpu": gpu, "enable_internet": internet,
            "dataset_sources": list(dataset_sources), "competition_sources": [], "kernel_sources": list(kernel_sources)}
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    return kdir
