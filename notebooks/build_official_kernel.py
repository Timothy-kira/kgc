"""Kaggle CPU kernel: import official daily episode datasets into replay-DB shards (data/import_official.py).

python notebooks/build_official_kernel.py <kernel_dir> <owner> <idx> <date,date,...>
Mounts xishengfeng/kaggriculture-replay-db (skip list, team ids, ratings), kaggle/kaggriculture-episodes-index
(manifest) and kaggle/kaggriculture-episodes-<date> for each date. Output: /kaggle/working/official/shards/.
"""
import sys

from kernel_common import DEFAULT_FILES, build_kernel

BODY = r'''
import glob, json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments==1.32.7"], check=False)
dbs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/state.json", recursive=True)
       if os.path.isdir(os.path.join(os.path.dirname(p), "index"))]
man = (glob.glob("/kaggle/input/**/kaggriculture-episodes-index/**/manifest.csv", recursive=True)
       or glob.glob("/kaggle/input/**/kaggriculture-episodes-index/manifest.csv", recursive=True) or [""])[0]
print("DB", dbs, "manifest", man, flush=True)
if not dbs:
    sys.exit("no replay DB under /kaggle/input")
days = sorted(glob.glob("/kaggle/input/**/kaggriculture-episodes-20*", recursive=True))
print("day dirs", len(days), days[:3], flush=True)
subprocess.run([sys.executable, "-m", "data.import_official", "/kaggle/working/official", dbs[0],
                "/kaggle/input/**/kaggriculture-episodes-20*/**/*.json", str(os.cpu_count()), "0.01", man], check=True)
'''

if __name__ == "__main__":
    kdir, owner, idx, dates = sys.argv[1:5]
    ds = ["xishengfeng/kaggriculture-replay-db", "kaggle/kaggriculture-episodes-index"] + \
         [f"kaggle/kaggriculture-episodes-{d}" for d in dates.split(",")]
    build_kernel(kdir, f"{owner}/kgc-official-{idx}", f"kgc official {idx}", BODY,
                 files=DEFAULT_FILES + ["data/import_official.py"], dataset_sources=ds)
