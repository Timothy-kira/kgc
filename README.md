# Kaggriculture RL

RL pipeline for the Kaggle *Kaggriculture* simulation competition, following
CodeMidas ("code itself as the RL environment") and MiMo-V2.6 (group-relative RL,
dynamic sampling, groupwise advantage redistribution, prefix-conditioned rollouts).

- `env/fast_env.py` – fast wrapper around the official interpreter (bit-identical game logic)
- `env/replay_check.py` – execution-consistency check vs. online replays
- `data/crawl.py` – resumable large-scale replay crawler -> compact DB (seed + actions)
- `data/replay_db.py` – DB reader + exact re-simulation
- `notebooks/build_replay_db_kernel.py` – builds the public Kaggle notebook that maintains the DB
- `league/` – public agents used as fixed league opponents (Apache-2.0, see file headers)
- `tools/arena.py` – round-robin evaluation

Public replay DB: dataset `xishengfeng/kaggriculture-replay-db`,
notebook `xishengfeng/kaggriculture-replay-database`.
