"""G2: is the Engram opponent memory memorisation or generalisation?

python -m tools.eval_engram <ckpt_dir> <seq_glob> <opp_glob> <opp_vocab> <train_db_dir> <eval_db_dir> [max_batches=60]
Per prediction horizon, opponent-event accuracy with Engram on / gates closed, against the majority-bucket
baseline, split by whether the opponent's team appears anywhere in the training replay DB ("seen") or not.
"""
import json
import sys

import numpy as np
import torch

from agent.opp_events import HORIZONS
from data.replay_db import ReplayDB
from model.clm_policy import CLMPolicy
from model.dsv41 import ModelArgs
from model.opp_data import LAYER_D, LAYER_S, OppIndex
from model.seq_batch import obs_feats


def main():
    ck, seq_glob, opp_glob, vocab, train_db, eval_db = sys.argv[1:7]
    max_b = int(sys.argv[7]) if len(sys.argv) > 7 else 60
    d = json.load(open(f"{ck}/model_args.json"))
    args = ModelArgs(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in ModelArgs.__dataclass_fields__})
    net = CLMPolicy(args)
    net.load_state_dict(torch.load(f"{ck}/engram.pt", map_location="cpu"), strict=False)
    net.eval()
    te = ReplayDB(train_db).episodes(columns=["episode_id", "team_id_0", "team_id_1"])
    seen = set(te.team_id_0) | set(te.team_id_1)
    ev = ReplayDB(eval_db).episodes(columns=["episode_id", "team_id_0", "team_id_1"]).set_index("episode_id")
    opp = OppIndex(opp_glob, vocab)
    import glob
    import random
    from data.seq_extract import load_trajs
    from model.clm_batch import collate
    from model.train_clm import crop
    rng = random.Random(3)
    stats = {g: {m: [[0, 0] for _ in HORIZONS] for m in ("engram", "closed", "majority")} for g in ("seen", "unseen")}
    counts = np.zeros((len(HORIZONS), 9, 9))
    n_b = 0

    def batches():
        for f in sorted(glob.glob(seq_glob)):
            for t in load_trajs(f):
                if t["episode_id"] in ev.index and opp.attach(t):
                    yield t["episode_id"], t["seat"], collate([crop(t, 240, rng)], device="cpu")

    with torch.no_grad():
        for e_id, seat, b in batches():
            row = ev.loc[e_id]
            opp_team = row[f"team_id_{1 - seat}"]
            g = "seen" if opp_team in seen else "unseen"
            feats = obs_feats(b, torch.float32)
            for mode in ("engram", "closed"):
                net.engram_ids = {LAYER_S: b["eng_s"], LAYER_D: b["eng_d"]}
                net.engram_mask = b["eng_mask"] if mode == "engram" else torch.zeros_like(b["eng_mask"])
                net.forward_train(b["ids"], feats, b["otype"], b["dec_pos"], b["dec_slot"], b["dec_desc"], b["tgt_pos"],
                                  b["tgt_kind"], b["val_pos"], b["mtp_pos"], b["mtp_kind"], None, b["opp_pos"])
                pred = net.last_opp[0].argmax(-1)
                ok = (pred == b["opp_tgt"]).float()
                for h in range(len(HORIZONS)):
                    stats[g][mode][h][0] += float(ok[:, h].sum())
                    stats[g][mode][h][1] += ok[:, h].numel()
            tg = b["opp_tgt"].numpy()
            for h in range(len(HORIZONS)):
                for p in range(9):
                    counts[h, p] += np.bincount(tg[:, h, p], minlength=9)
                    stats[g]["majority"][h][1] += len(tg)
            n_b += 1
            if n_b >= max_b:
                break
    maj = counts.argmax(-1)                     # majority bucket per (horizon, product), from the eval data itself
    print(json.dumps({"batches": n_b}))
    for g in ("seen", "unseen"):
        for mode in ("engram", "closed"):
            accs = [round(a / max(1, n), 3) for a, n in stats[g][mode]]
            print(g, mode, dict(zip([str(h) for h in HORIZONS], accs)))
    base = [round(float(counts[h].max(-1).sum() / counts[h].sum()), 3) for h in range(len(HORIZONS))]
    print("majority (all eval)", dict(zip([str(h) for h in HORIZONS], base)))


if __name__ == "__main__":
    main()
