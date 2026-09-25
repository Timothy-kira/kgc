# Kaggriculture v3: RL-controllable heuristic experts + multi-teacher on-policy distillation

## Why change the structure
- The pure imitation policy (CSA2 decoder + CLM heads, pre-trained on 20k replays) learned the **opening**
  (visible once decoding is hierarchical: op → item → qty), but it cannot run the farming loop
  (plant → water → harvest → sell with 10+ hands): unit-level agreement with top players is only ~60%,
  errors compound, and it ends every game with ~0 money (0–70 vs 130k–190k for public agents).
- The leaderboard top (~3050) is heuristic agents plus some tuning/RL. Heuristics give flawless execution;
  what separates near-mirror top games (winner ahead by ~2.5%) is **strategy**: crop/animal portfolio against
  the rival's production, land/hire timing, sell timing under the shared market price curves.
- Local env speed: a heuristic-vs-heuristic game takes **~2 s** (720 steps), so ~7k games/hour on 4 CPU cores.
  RL at scale is feasible on CPU; GPU is optional.

## Architecture: Expert-Routed Policy (ERP)
```
obs ──► CSA2 decoder backbone (pre.pt: replay-pretrained obs encoder + action priors)
          │  per step: 33 obs tokens + expert-proposal tokens + <ACT>, then slot decisions
          ▼
   ┌──────────── slot decision (CLM head over a *dynamic* candidate set) ────────────┐
   │ candidates(slot) = { proposal of expert 1..K for this slot }  ∪  closed set    │
   │ farmer / each hand / each market order: pick an expert's decision or deviate   │
   └─────────────────────────────────────────────────────────────────────────────────┘
          │  once per day: knob head  → settings of the parameterised experts
          ▼
   K heuristic experts (run every step, ~1.5 ms each):
     metav4, farm2945, cha22, v48, salem2900 (public, strongest local: metav4)
     + parameterised variants: DEFAULT_SETTINGS / market params exposed as knobs
       (sell_lead, front_run, block_turns, min_sell_price, crop/animal portfolio targets,
        hire cap, land timing, sell fraction per product)
```
- **Heuristics are controllable, not fixed**: (1) knob head sets expert parameters each day (RL-tuned);
  (2) the router picks, per slot, which expert's decision to execute; (3) the policy can deviate from all
  experts with its own candidate (e.g. a better sell timing).
- CLM heads fit this naturally: scores are cosine similarities with candidate encodings, so expert proposals
  are just extra candidates (encoded with the same action encoder + an expert-id embedding).
- Experts see the real executed history (they are called every step with the true observation); a
  consistency check guards experts whose internal plan state assumes their own actions were executed.

## Training (pre → mid → post)
1. **Pre (done)**: replay distillation → `pre.pt` (obs encoder, opening, action priors).
2. **Mid = Multi-teacher On-Policy Distillation (MOPD, MiMo-V2 style)**
   - The student plays in the local env against the league; at every *student-visited* state all K experts are
     queried (cheap), giving per-slot teacher targets. Loss: reverse-KL / CE to a teacher mixture on the
     student's own states (DAgger-like: no compounding error, fixes the closed-loop failure of pure BC).
   - Teacher weights per state = advantage of each expert from that state (short rollouts in the exact env:
     "which expert does best from here"), so the student learns *when* each expert is best (routing).
   - Replay BC kept as a small regulariser (top-player strategy not covered by any expert).
3. **Post = RL (GRPO, MiMo eq.1 / CodeMidas advantages)**
   - Actions: router choices + daily knob settings + free deviations. Reward: win + 0.25·tanh(diff/15000).
   - Opponents: public experts, self-play snapshots (PFSP), exploiters trained against the current policy
     (dynamic adversaries). KL to the MOPD policy.
   - Rollouts on CPU workers (Kaggle CPU kernels in parallel + local); learner on GPU only if needed.

## Decoding / inference budget (1 s/step, 2 threads)
- Experts ~5–10 ms total; policy decode ~100 ms with hierarchical selection; knob head once per day.
- Fallback: if the policy is late or errors, execute the best expert's full action (never forfeit).

## Schedule (deadline 2026-09-30; ratings need days of games → submit early)
| Day | Work | Submission |
|---|---|---|
| 09-25 | Expert wrapper: run K experts per step, per-slot proposals, knob exposure; verify "route-all-to-metav4" == metav4 exactly; oracle study: per-game best expert / per-day switching gain | — |
| 09-26 | Router + knob head on the CSA2 backbone; MOPD data loop (student rollouts + expert queries) on CPU; quick routing-only RL (CEM/GRPO over knobs) | v3a: best knob-tuned expert |
| 09-27 | MOPD mid-train (GPU ≤6 h); eval vs league both seats | v3b: ERP after MOPD |
| 09-28 | GRPO post-train with league + self-play + exploiters | v3c |
| 09-29 | Final eval (≥100 games, both seats, all experts), latency check, final submission | v3d |

## Verification
- Wrapper: routing everything to expert k reproduces expert k's games bit-exactly.
- Each stage: closed-loop win rate / money vs every expert (both seats, ≥50 games), latency (p99, max),
  never-forfeit test; opening probe (`tools/opening_probe.py`) for qualitative checks.
