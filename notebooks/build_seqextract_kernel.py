"""Kaggle CPU kernel: convert a range of replay-DB shards into decoder-only training sequences.

python notebooks/build_seqextract_kernel.py <kernel_dir> <owner> <idx> <shard_from> <shard_to>
Output: /kaggle/working/seq/seq_ep_XXXXX.npz (input for the GPU pre-training kernel)
"""
import sys

from kernel_common import build_kernel

BODY = r'''
import glob, json
dbs = [p for p in glob.glob("/kaggle/input/**/state.json", recursive=True) if os.path.isdir(os.path.join(os.path.dirname(p), "index"))]
db = os.path.dirname(max(dbs, key=lambda p: json.load(open(p)).get("n_downloaded", 0)))
print("DB", db, flush=True)
ncpu = os.cpu_count()
subprocess.run([sys.executable, "-m", "data.seq_extract", db, "/kaggle/working/seq", "1800", str(ncpu), "__RANGE__"],
               check=True)
print("files", len(glob.glob("/kaggle/working/seq/*.npz")))
'''

if __name__ == "__main__":
    kdir, owner, idx, a, b = sys.argv[1:6]
    build_kernel(kdir, f"{owner}/kgc-seqextract-{idx}", f"kgc seqextract {idx}", BODY.replace("__RANGE__", f"{a}:{b}"),
                 dataset_sources=["xishengfeng/kaggriculture-replay-db"])
