"""Package the CLM-head policy as a Kaggle submission (submission.tar.gz with main.py at the root).

python -m submit.pack_clm <weights.pt> <model_args.json> <out_dir> [temperature]
The archive contains main.py, the inference sources and the weights. Runtime: torch 2.6 CPU, 2 threads.
"""
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["agent/__init__.py", "agent/action_space.py", "agent/action_tokens.py", "agent/features.py",
         "agent/obs_tokens.py", "agent/clm_agent.py", "model/__init__.py", "model/dsv41.py", "model/clm_policy.py"]

MAIN = r'''# Kaggriculture agent: decoder-only (DeepSeek-V4.1-style CSA2) policy with CLM action heads.
import os, sys, time
_CANDS = ([os.path.dirname(os.path.abspath(__file__))] if "__file__" in globals() else []) + \
         ["/kaggle_simulations/agent", os.getcwd(), os.environ.get("KGC_AGENT_DIR", "")]
HERE = next((d for d in _CANDS if d and os.path.exists(os.path.join(d, "weights.pt"))), "/kaggle_simulations/agent")
sys.path.insert(0, HERE)
import torch
torch.set_num_threads(2)
from agent.clm_agent import CLMAgent, load_clm

_AGENT = None
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def agent(observation, configuration=None):
    global _AGENT
    if _AGENT is None:
        _AGENT = CLMAgent(load_clm(os.path.join(HERE, "weights.pt"), os.path.join(HERE, "model_args.json")),
                          temperature=__TEMP__, time_budget=0.8)
    try:
        return _AGENT(observation, configuration)
    except Exception as e:   # never forfeit a game on an unexpected error
        print("agent error", repr(e), flush=True)
        return PASS
'''


def build(weights, args_json, out_dir, temperature=0.0):
    stage = os.path.join(out_dir, "pkg")
    shutil.rmtree(stage, ignore_errors=True)
    for rel in FILES:
        os.makedirs(os.path.join(stage, os.path.dirname(rel)), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), os.path.join(stage, rel))
    shutil.copy(weights, os.path.join(stage, "weights.pt"))
    shutil.copy(args_json, os.path.join(stage, "model_args.json"))
    open(os.path.join(stage, "main.py"), "w").write(MAIN.replace("__TEMP__", repr(float(temperature))))
    tar = os.path.join(out_dir, "submission.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        for f in sorted(os.listdir(stage)):
            t.add(os.path.join(stage, f), arcname=f)
    return tar, stage


if __name__ == "__main__":
    tar, stage = build(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    print(tar, os.path.getsize(tar))
