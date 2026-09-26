"""Kaggle CPU kernel: seq + opponent-event extraction of a shard range, for Engram (re)training on all replays.

python notebooks/build_engdata_kernel.py <kernel_dir> <owner> <idx> <shard_from> <shard_to> [min_score=1500] [extra_kernel ...]
Resumes from the public replay DB (state.json at the dataset root) plus, optionally, crawl kernel outputs
(kgc-crawl-k: crawl_out/shards/ep_*_p<k>.parquet) merged into one shard list.
Output: /kaggle/working/seq/seq_ep_*.npz (data/seq_extract.py) and /kaggle/working/opp/ep_*.npz (data/opp_flow_extract.py).
"""
import sys

from kernel_common import DEFAULT_FILES, build_kernel

BODY = r'''
import glob, json, shutil
dbs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/state.json", recursive=True)
       if os.path.isdir(os.path.join(os.path.dirname(p), "index"))]
dbs.sort(key=lambda d: -json.load(open(os.path.join(d, "state.json"))).get("n_downloaded", 0))
print("DBs", dbs, flush=True)
if not dbs:
    subprocess.run("find /kaggle/input -maxdepth 4 | head -50", shell=True)
    sys.exit("no replay DB under /kaggle/input")
db = dbs[0]
extra = [f for f in glob.glob("/kaggle/input/**/crawl_out/shards/ep_*_p*.parquet", recursive=True)]
if extra:                                   # merged view: symlinks to the base shards + crawl shards
    m = "/kaggle/tmp_db"
    os.makedirs(m + "/shards", exist_ok=True)
    for f in glob.glob(db + "/shards/*.parquet") + extra:
        dst = os.path.join(m, "shards", os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(f, dst)
    shutil.copytree(db + "/index", m + "/index")
    shutil.copy(db + "/state.json", m)
    db = m
print("DB", db, "shards", len(glob.glob(db + "/shards/*.parquet")), "extra", len(extra), flush=True)
ncpu = str(os.cpu_count())
subprocess.run([sys.executable, "-m", "data.opp_flow_extract", db, "/kaggle/working/opp", ncpu, "__RANGE__"], check=True)
subprocess.run([sys.executable, "-m", "data.seq_extract", db, "/kaggle/working/seq", "__MS__", ncpu, "__RANGE__"],
               check=True)
print("seq files", len(glob.glob("/kaggle/working/seq/*.npz")), "opp files", len(glob.glob("/kaggle/working/opp/*.npz")))
'''

if __name__ == "__main__":
    kdir, owner, idx, a, b = sys.argv[1:6]
    ms = sys.argv[6] if len(sys.argv) > 6 else "1500"
    extra = sys.argv[7:]
    build_kernel(kdir, f"{owner}/kgc-engdata-{idx}", f"kgc engdata {idx}",
                 BODY.replace("__RANGE__", f"{a}:{b}").replace("__MS__", ms),
                 files=DEFAULT_FILES + ["data/opp_flow_extract.py"],
                 dataset_sources=["xishengfeng/kaggriculture-replay-db"], kernel_sources=extra)
