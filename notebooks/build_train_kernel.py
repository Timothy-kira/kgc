"""Kaggle GPU kernel: pre-train / mid-train the decoder-only policy on extracted replay sequences.

python notebooks/build_train_kernel.py <kernel_dir> <owner> <data_dataset> <stage> [hours] [init_kernel]
Output: /kaggle/working/ckpt/{stage}.pt, model_args.json, train.log
"""
import sys

from kernel_common import build_kernel

BODY = r'''
import glob, subprocess, sys
os.environ["PYTHONUNBUFFERED"] = "1"          # live logs: `kaggle kernels logs -f <slug>`
files = glob.glob("/kaggle/input/**/seq_*.npz", recursive=True)
print("seq files", len(files), flush=True)
subprocess.run(["nvidia-smi"], check=False)
init = glob.glob("/kaggle/input/**/ckpt/__INIT__.pt", recursive=True)
import torch
ngpu = max(1, torch.cuda.device_count())
launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}"]
            if ngpu > 1 else [sys.executable, "-X", "faulthandler", "-u"])
cmd = launcher + ["-m", "__MODULE__", "--data", ",".join(sorted(set(os.path.dirname(f) for f in files))
       and [d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files))]),
       "--out", "/kaggle/working/ckpt", "--stage", "__STAGE__", "--max_hours", "__HOURS__",
       "--batch", "__BATCH__", "--crop_steps", "__CROP__", "--epochs", "__EPOCHS__"] + "__EXTRA__".split()
if init:
    cmd += ["--init", init[0]]
print(" ".join(cmd), flush=True)

# ---- monitors (appear in the live log `kaggle kernels logs -f`):
#  * GPU utilisation / memory every 60 s
#  * inference latency vs. competition limits (CPU, 2 threads, full game) for every new checkpoint
import threading, time as _time
def _gpu_monitor():
    while True:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        print("GPU_MON", r.stdout.strip().replace("\n", " | "), flush=True)
        _time.sleep(60)
def _latency_monitor():
    last = 0.0
    ck = "/kaggle/working/ckpt/__STAGE__.pt"
    while True:
        _time.sleep(120)
        if os.path.exists(ck) and os.path.getmtime(ck) > last + 1:
            last = os.path.getmtime(ck)
            r = subprocess.run([sys.executable, "-m", "tools.bench_latency", ck, "/kaggle/working/ckpt/model_args.json",
                                "719", "2"], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
            print("LATENCY", (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1], flush=True)
threading.Thread(target=_gpu_monitor, daemon=True).start()
threading.Thread(target=_latency_monitor, daemon=True).start()
with open("/kaggle/working/train.log", "w") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "open_spiel" in line:                  # kaggle_environments import noise
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    rc = p.wait()
    print("train exit code", rc, flush=True)
    log.write(f"train exit code {rc}\n")
subprocess.run("free -g; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv", shell=True)
'''


def build(kdir, owner, data_ds, stage, hours="10.5", init="pre", batch="4", crop="240", epochs="3", slug=None,
          init_kernel=None, extra="", module="model.train_clm"):
    body = (BODY.replace("__MODULE__", module).replace("__EXTRA__", extra).replace("__STAGE__", stage).replace("__HOURS__", hours).replace("__INIT__", init)
            .replace("__BATCH__", batch).replace("__CROP__", crop).replace("__EPOCHS__", epochs))
    body = body.replace('",".join(sorted(set(os.path.dirname(f) for f in files))\n       and [d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files))])',
                        '",".join(d + "/seq_*.npz" for d in sorted(set(os.path.dirname(f) for f in files)))')
    return build_kernel(kdir, slug or f"{owner}/kgc-train-{stage}", f"kgc train {stage}", body, gpu=True,
                        dataset_sources=[data_ds], kernel_sources=[init_kernel] if init_kernel else [])


if __name__ == "__main__":
    build(*sys.argv[1:])
