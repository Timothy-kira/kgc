"""Build the public Kaggle kernel that maintains the Kaggriculture replay database.

The generated kernel script embeds the crawler / reader / env sources, resumes
from the newest database it can find among its inputs (the public dataset
and/or this kernel's previous output), keeps crawling for up to MAX_HOURS and
leaves the updated database in /kaggle/working/replay_db.

    python notebooks/build_replay_db_kernel.py <out_dir> <owner>
    kaggle kernels push -p <out_dir>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["data/__init__.py", "data/crawl.py", "data/replay_db.py", "env/__init__.py", "env/fast_env.py",
         "env/replay_check.py"]

TEMPLATE = r'''# Kaggriculture Replay Database (resumable crawler)
#
# A public, reusable database of Kaggriculture ladder replays.
# Each episode is stored compactly as seed + both players' per-step actions
# (+ ratings, final money, daily market snapshots). The game is deterministic,
# so `ReplayDB.rebuild(episode_id)` re-simulates every state exactly with the
# official interpreter (verified step-by-step against online replays).
#
# Resume: on every run the crawler continues from the newest database found
# in /kaggle/input (the public dataset and/or this notebook's previous output),
# so re-running only fetches games that are not in the database yet.
#
# Use it:  add this notebook's output (or the dataset
#   {owner}/kaggriculture-replay-db) as input, then
#     import sys; sys.path.insert(0, "/kaggle/input/<name>/src")
#     from data.replay_db import ReplayDB
#     db = ReplayDB("/kaggle/input/<name>/replay_db")
#     df = db.episodes(min_score=2800)
import glob, json, os, subprocess, sys, time

MAX_HOURS = float(os.environ.get("MAX_HOURS", "11.3"))
MIN_SCORE = os.environ.get("MIN_SCORE", "1800")
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC = os.path.join(WORK, "src")
OUT = os.path.join(WORK, "replay_db")

for pkg in ("zstandard", "pyarrow", "kaggle_environments"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        pkg.replace("_", "-")], check=False)

FILES = {files}
for rel, text in FILES.items():
    p = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
sys.path.insert(0, SRC)

# ---- pick the newest resume source among inputs
best, best_n = None, -1
for st in glob.glob("/kaggle/input/**/replay_db/state.json", recursive=True):
    try:
        n = json.load(open(st)).get("n_downloaded", 0)
    except Exception:
        continue
    if n > best_n:
        best, best_n = os.path.dirname(st), n
print("resume source:", best, "episodes:", best_n, flush=True)

args = ["--out", OUT, "--min-score", MIN_SCORE, "--max-hours", str(MAX_HOURS), "--rounds", "1000",
        "--max-lists", "200", "--workers", "12", "--list-workers", "6", "--shard-size", "250",
        "--verify-fraction", "0.01", "--seed-subs", "{seed_subs}"]
if best:
    args += ["--resume-from", best]
from data import crawl
crawl.main(args)

# ---- summary
from data.replay_db import ReplayDB
db = ReplayDB(OUT)
df = db.episodes()
print(df[["updated_score_0", "updated_score_1", "reward_0", "reward_1"]].describe())
with open(os.path.join(WORK, "README.md"), "w") as f:
    f.write("# Kaggriculture replay database\n\n"
            f"episodes: {{len(df)}}  updated: {{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}}\n\n"
            "See the notebook header for usage.\n")
'''


def build(out_dir, owner, seed_subs=""):
    os.makedirs(out_dir, exist_ok=True)
    files = {rel: open(os.path.join(ROOT, rel)).read() for rel in FILES}
    code = TEMPLATE.replace("{owner}", owner).replace("{seed_subs}", seed_subs)
    code = code.replace("{{", "{").replace("}}", "}").replace("{files}", repr(files))
    with open(os.path.join(out_dir, "kaggriculture_replay_db.py"), "w") as f:
        f.write(code)
    slug = f"{owner}/kaggriculture-replay-database"
    meta = {
        "id": slug, "title": "Kaggriculture Replay Database", "code_file": "kaggriculture_replay_db.py",
        "language": "python", "kernel_type": "script", "is_private": False,
        "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [f"{owner}/kaggriculture-replay-db"],
        "competition_sources": [], "kernel_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
