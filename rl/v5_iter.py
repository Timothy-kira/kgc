"""Large-scale iterated policy improvement for the A track on one big box (docs/PLAN_RL.md, section 12).

python -m rl.v5_iter --out DIR --pools rl_pools.pkl --eval_pools rl_pools_v2.pkl --vocab opp_vocab.npz
       --seed_samples '<glob,glob>' --engram engram.pt [--base_policy pi.pt --base_rule 0.25/2.0] [--hours 11]
       [--round_min 90] [--procs 0]

Round 0  (GPU): rl.v5_offline on all seed samples (earlier sampler kernels), tables from --engram, ladder weighting
         -> pi_0 (+ its acting rule, chosen on the ladder-weighted held-out gain with near/top >= 0).
Round k  (CPU, all cores): rl.v5_rl --sample_only acting with pi_{k-1} (its rule); opponents = replay tapes
         (near/top/loss), live league agents, v4b mirrors and self-play against pi_{k-1} ("selfpi"); every alternative
         of one head is rolled out exactly at key decisions (Q^{pi_{k-1}} differences).
         (GPU) rl.v5_offline warm-started from pi_{k-1} on the last two rounds' samples (+ seed), ladder weighting.
         (CPU) tools.v5_recheck: 313 fresh paired games vs v4b; the candidate is kept only if its ladder-weighted
         win difference beats the incumbent's (the incumbent starts as --base_policy when given).
Everything lands in DIR: round_k/{samples,learn,recheck.json}, best.json (policy path + rule + recheck).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


def run(cmd, log):
    print("RUN", " ".join(cmd), flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "open_spiel" in line:
                continue
            f.write(line)
            if line.startswith(("EPOCH", "BEST", "ITER", "{", "EVAL", "init", "eng_init")):
                print(line, end="", flush=True)
        return p.wait()


def recheck(policy, rule, a, out):
    rc = run([sys.executable, "-u", "-m", "tools.v5_recheck", policy, a.eval_pools, a.vocab, out, rule, str(a.procs)],
             out + ".log")
    if rc != 0 or not os.path.exists(out):
        return None
    r = json.load(open(out))
    recs = [v for v in r.values() if isinstance(v, dict)]
    return max(recs, key=lambda v: v["wdwin"]) if recs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--eval_pools", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--seed_samples", default="")
    ap.add_argument("--engram", default="")
    ap.add_argument("--base_policy", default="")
    ap.add_argument("--base_rule", default="0.25/2.0")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--round_min", type=float, default=90.0)
    ap.add_argument("--learn_hours", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--selfpi", type=float, default=0.2, help="share of self-play games against the current policy")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reserve_min", type=float, default=40.0, help="time kept free per round for learning + recheck")
    a = ap.parse_args()
    t_end = time.time() + a.hours * 3600
    os.makedirs(a.out, exist_ok=True)
    if a.procs <= 0:
        a.procs = max(2, (os.cpu_count() or 4) - 2)
    print(json.dumps({"cpus": os.cpu_count(), "procs": a.procs}), flush=True)
    py = sys.executable
    best = None
    if a.base_policy:
        rec = recheck(a.base_policy, a.base_rule, a, os.path.join(a.out, "base_recheck.json"))
        if rec:
            best = {"policy": a.base_policy, "rule": a.base_rule, "recheck": rec}
            print("BASE", json.dumps({"wdwin": rec["wdwin"], "win": rec["win"], "by_kind": rec["by_kind"]}), flush=True)

    def learn(k, samples, init):
        d = os.path.join(a.out, f"round_{k}", "learn")
        cmd = [py, "-u", "-m", "rl.v5_offline", "--samples", samples, "--vocab", a.vocab, "--out", d, "--weights",
               "ladder", "--hours", str(a.learn_hours), "--epochs", "40", "--bs", "2048", "--freeze_tables"]
        cmd += ["--init_policy", init] if init else (["--eng_init", a.engram] if a.engram else [])
        run(cmd, d + ".log")
        pj = os.path.join(d, "policy.json")
        if not os.path.exists(pj):
            return None
        pol = json.load(open(pj))
        return os.path.join(d, "policy.pt"), f"{pol['margin']}/{pol['c']}", pol

    def consider(k, policy, rule):
        nonlocal best
        rec = recheck(policy, rule, a, os.path.join(a.out, f"round_{k}", "recheck.json"))
        if rec is None:
            return
        print("RECHECK", k, json.dumps({"rule": rule, "wdwin": rec["wdwin"], "se": rec["wdwin_se"], "win": rec["win"],
                                         "by_kind": rec["by_kind"]}), flush=True)
        if best is None or rec["wdwin"] > best["recheck"]["wdwin"]:
            best = {"policy": policy, "rule": rule, "recheck": rec, "round": k}
            json.dump(best, open(os.path.join(a.out, "best.json"), "w"), indent=1)
            print("NEW_BEST", k, flush=True)

    # round 0: learner on the seed samples with the (new) Engram tables
    k = 0
    cur = None
    if a.seed_samples:
        os.makedirs(os.path.join(a.out, "round_0"), exist_ok=True)
        r = learn(0, a.seed_samples, "")
        if r:
            consider(0, r[0], r[1])
    cur = (best["policy"], best["rule"]) if best else (None, a.base_rule)
    rounds = []
    while time.time() + (a.round_min + a.learn_hours * 60 + a.reserve_min) * 60 < t_end and cur[0]:
        k += 1
        rd = os.path.join(a.out, f"round_{k}")
        os.makedirs(rd, exist_ok=True)
        m, c = cur[1].split("/")[:2]
        smp = os.path.join(rd, "samples")
        run([py, "-u", "-m", "rl.v5_rl", "--pools", a.pools, "--vocab", a.vocab, "--out", smp, "--hours",
             str(a.round_min / 60), "--procs", str(a.procs), "--obj", "adv", "--ens", "5", "--sample_only",
             "--batch", str(a.batch), "--init_policy", cur[0], "--margin", m, "--lcb_c", c,
             "--mix", f"selfpi={a.selfpi}"], os.path.join(rd, "sample.log"))
        rounds.append(smp)
        pats = ",".join(os.path.join(s, "samples_*.pkl") for s in rounds[-2:])
        if a.seed_samples:
            pats += "," + a.seed_samples
        r = learn(k, pats, cur[0])
        if r:
            consider(k, r[0], r[1])
        if best:
            cur = (best["policy"], best["rule"])
    print("DONE", json.dumps(best["recheck"] if best else None), flush=True)


if __name__ == "__main__":
    main()
