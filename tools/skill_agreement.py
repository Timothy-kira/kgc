"""How well do the decomposed heuristic skills explain top players' decisions?

python -m tools.skill_agreement <replay_db_dir> <team_name_substring> [n_episodes=3] [experts=...]
Re-simulates top-team episodes, runs the skill pool in shadow mode on the top player's observations, and
reports per slot family (unit / market) the fraction of decisions where (a) any skill proposes exactly the
player's decision ("oracle gate" coverage) and (b) the single best skill does; plus the best skills.
"""
import collections
import json
import sys

import pandas as pd

from agent.action_space import SLOT_MARKET, STOP, action_to_decisions
from agent.skills import SkillPool, market_proposal, slot_proposals
from data.replay_db import ReplayDB, _unz

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def main():
    db_dir, team = sys.argv[1], sys.argv[2]
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    names = (sys.argv[4] if len(sys.argv) > 4 else "metav4,farm2945,cha22,v48,salem2900").split(",")
    db = ReplayDB(db_dir)
    e = db.episodes()
    rows = []
    for k in (0, 1):
        m = e["team_names"].map(lambda x: team in (json.loads(x)[k] if isinstance(x, str) else x[k]))
        rows.append(e[m].assign(seat=k))
    sel = pd.concat(rows).sort_values("episode_id").tail(n_ep)
    tot = {"unit": [0, 0, 0], "market": [0, 0, 0]}          # any-skill hits, decisions, (unused)
    per_skill = {"unit": collections.Counter(), "market": collections.Counter()}
    for _, r in sel.iterrows():
        pool = SkillPool([f"league/{n}.py" for n in names])
        row = db.row(int(r.episode_id))
        acts = _unz(row["actions_zstd"])
        seat = int(r.seat)
        for t, env in ReplayDB.rebuild(None, int(r.episode_id), row=row):
            if t >= len(acts) or env.done:
                break
            obs = env.obs(seat)
            n_hands = len(obs["farms"][seat]["hands"])
            props = [a for _, a in pool.propose(obs, env.config)]
            units, markets = slot_proposals(props, n_hands)
            a = acts[t][seat] if isinstance(acts[t][seat], dict) else PASS
            dec = action_to_decisions(a, n_hands)
            j = 0
            for pos, (slot, c) in enumerate(dec):
                if slot == SLOT_MARKET:
                    p = market_proposal(markets, j)
                    j += 1
                    fam = "market"
                else:
                    p = [u[pos] for u in units]
                    fam = "unit"
                hit = [i for i, x in enumerate(p) if x == c]
                tot[fam][0] += bool(hit)
                tot[fam][1] += 1
                for i in hit:
                    per_skill[fam][pool.names[i]] += 1
        print(r.episode_id, seat, {f: round(v[0] / max(v[1], 1), 3) for f, v in tot.items()}, flush=True)
    for f in ("unit", "market"):
        n = max(tot[f][1], 1)
        best = per_skill[f].most_common(5)
        print(f, "any-skill coverage", round(tot[f][0] / n, 3), "best single skills",
              [(k, round(v / n, 3)) for k, v in best])


if __name__ == "__main__":
    main()
