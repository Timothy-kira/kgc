"""Keep exactly ONE of our GPU notebooks alive on the GPU account.

Policy: before pushing a new GPU notebook, every existing kgc GPU kernel (slug prefix `kgc-train` /
`kgc-gpu`) is deleted (its outputs are downloaded first, so checkpoints are never lost). Other notebooks on
the account are never touched.

    python -m tools.kaggle_gpu replace <kernel_dir> [--keep-outputs DIR]    # delete old kgc GPU kernels, push new
    python -m tools.kaggle_gpu clean [--keep-outputs DIR]                   # delete old kgc GPU kernels only
    python -m tools.kaggle_gpu quota
Credentials: KAGGLE_API_TOKEN of the GPU account.
"""
import json
import os
import re
import subprocess
import sys

PREFIXES = ("kgc-train", "kgc-gpu")


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def my_kernels():
    out = _run(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]).stdout.splitlines()
    refs = [l.split(",")[0] for l in out[1:] if l.strip()]
    return [r for r in refs if r.split("/")[-1].startswith(PREFIXES)]


def clean(keep_outputs=None):
    for ref in my_kernels():
        if keep_outputs:
            d = os.path.join(keep_outputs, ref.replace("/", "__"))
            os.makedirs(d, exist_ok=True)
            r = _run(["kaggle", "kernels", "output", ref, "-p", d], timeout=3600)
            print("saved outputs of", ref, "->", d, r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-200:])
        r = _run(["kaggle", "kernels", "delete", ref, "-y"])
        print("deleted", ref, (r.stdout or r.stderr).strip()[-200:])


def replace(kernel_dir, keep_outputs=None):
    meta = json.load(open(os.path.join(kernel_dir, "kernel-metadata.json")))
    assert meta["id"].split("/")[-1].startswith(PREFIXES), "GPU kernels must use a kgc-train*/kgc-gpu* slug"
    clean(keep_outputs)
    r = _run(["kaggle", "kernels", "push", "-p", kernel_dir])
    print((r.stdout or r.stderr).strip()[-300:])


def quota():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        r = k.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
    print(r.to_json() if hasattr(r, "to_json") else r)


if __name__ == "__main__":
    cmd = sys.argv[1]
    keep = sys.argv[sys.argv.index("--keep-outputs") + 1] if "--keep-outputs" in sys.argv else None
    if cmd == "replace":
        replace(sys.argv[2], keep)
    elif cmd == "clean":
        clean(keep)
    elif cmd == "quota":
        quota()
