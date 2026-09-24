"""Kaggle GPU kernel: pre-train / mid-train the decoder-only policy on extracted replay sequences.

python notebooks/build_train_kernel.py <kernel_dir> <owner> <data_dataset> <stage> [hours] [init_kernel]
Output: /kaggle/working/ckpt/{stage}.pt, model_args.json, train.log
"""
import sys

from kernel_common import build_kernel

BODY = r'''
import glob, subprocess, sys
files = glob.glob("/kaggle/input/**/seq_*.npz", recursive=True)
print("seq files", len(files), flush=True)
subprocess.run(["nvidia-smi"], check=False)
init = glob.glob("/kaggle/input/**/ckpt/__INIT__.pt", recursive=True)
cmd = [sys.executable, "-X", "faulthandler", "-u", "-m", "model.train_seq", "--data", ",".join(sorted(set(os.path.dirname(f) for f in files))
       and [d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files))]),
       "--out", "/kaggle/working/ckpt", "--stage", "__STAGE__", "--max_hours", "__HOURS__",
       "--batch", "__BATCH__", "--crop_steps", "__CROP__", "--epochs", "__EPOCHS__"] + "__EXTRA__".split()
if init:
    cmd += ["--init", init[0]]
print(" ".join(cmd), flush=True)
with open("/kaggle/working/train.log", "w") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = p.wait()
    print("train exit code", rc, flush=True)
    log.write(f"train exit code {rc}\n")
subprocess.run("free -g; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv", shell=True)
'''


def build(kdir, owner, data_ds, stage, hours="10.5", init="pre", batch="4", crop="240", epochs="3", slug=None,
          init_kernel=None, extra=""):
    body = (BODY.replace("__EXTRA__", extra).replace("__STAGE__", stage).replace("__HOURS__", hours).replace("__INIT__", init)
            .replace("__BATCH__", batch).replace("__CROP__", crop).replace("__EPOCHS__", epochs))
    body = body.replace('",".join(sorted(set(os.path.dirname(f) for f in files))\n       and [d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files))])',
                        '",".join(d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files)))')
    return build_kernel(kdir, slug or f"{owner}/kgc-train-{stage}", f"kgc train {stage}", body, gpu=True,
                        dataset_sources=[data_ds], kernel_sources=[init_kernel] if init_kernel else [])


if __name__ == "__main__":
    build(*sys.argv[1:])
