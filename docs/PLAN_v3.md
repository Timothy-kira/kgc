# Kaggriculture v3：启发式技能拆解 × DeepSeek-V4.1 式 MoE 动作层 × 关键步 RL

## Context
- **保留现有架构**：
  - CSA2 decoder（`model/dsv41.py`）；
  - CLM 投影头（`model/clm_policy.py`）；
  - 槽位式决策：农民、每个雇工、市场订单直到 STOP；
  - 分层输出：操作 → 物品 → 数量（`agent/clm_agent.py` 的 `_pick`）。
- **纯模仿的问题**：开局学会了，但经营循环里单位动作只有约 60% 与高手一致，误差累积，资金归零。公开启发式 agent 能到 13–19 万。
- **用户要求**：
  1. 把启发式算法一步步拆开，做成类似 MoE 的专家；
  2. 只在关键步骤做 RL，用来加速 RL；
  3. MoE 要对照 DeepSeek-V4.1-Flash 源码，有针对性地模仿它的设计。
- **已核对的 DeepSeek-V4.1-Flash `inference/model.py`**（第 792–904 行）：
  - `Gate`：`scores = sqrtsoftplus(W·x / gate_temp)`；`indices = topk(scores + bias)`，偏置只负责选专家；`weights = scores.gather(indices)`，topk>1 时归一化，再乘 `route_scale`（Flash 为 1.5）；
  - 图像 token 用单独的偏置 `bias_vl`；
  - `Expert`：SwiGLU，up 分支双侧截断，gate 分支只截上限（`swiglu_limit`）；
  - `MoE`：y = Σ 路由专家(x)·权重 + 共享专家(x)；
  - `get_moe_config(layer_id)`：不同层可以配置不同的专家数；
  - 训练时用无辅助损失（noaux_tc）方式更新偏置。我们的移植版 `model/dsv41.py` 的 Gate、Expert、MoE 已经与之一致。
- **本地环境**：启发式每步约 1.5 ms，一局约 2 秒，支持 `FarmEnv.clone`。截止日期 09-30。

## 1. 把启发式拆成细粒度技能专家（`agent/skills.py`）
- 来源：metav4 Chassis 的各层，以及公开 agent 的通用逻辑。每个技能是
  `skill(obs, mem, slot_ctx) -> 候选编号 或 None（不适用）`，
  开销是微秒到毫秒级。所有技能每一步都在影子模式下运行，维护各自的记忆。
- **单位槽技能（农民 / 雇工）**：
  - 跟随计划：按动作带或计划执行；
  - 收获最近的成熟作物、浇未浇水的作物、在空地上种最优作物；
  - 喂食、照料、收集产物（鸡蛋、牛奶、羊毛）、收集肥料和施肥；
  - 清除杂草；
  - 回棚子放下或取出物品；
  - 原地等待。
- **市场槽技能**：
  - 按计划下单；
  - 提前卖出（sell_lead）、抢在对手前卖出（front_run）、限制卖出量（clamp_sells）；
  - 清理积压库存、终局清仓；
  - 按作物计划买种子、买动物；
  - 雇人（考虑斐波那契递增的成本）；
  - 买地时机；
  - 结束下单（STOP）。
- **细粒度专家**：同一技能的不同参数（例如卖出阈值、雇工上限），算作不同的专家，参考 DeepSeekMoE 的细粒度专家设计。

## 2. MoE 动作层（`model/hmoe.py`，逐行仿照 DSV4.1 的 Gate 和 MoE）
```
槽位隐状态 h ──► HGate：scores = sqrtsoftplus(W_g·h / gate_temp)
                  不适用的技能：scores 置为 -inf，不参与选择
                  indices = topk(scores + bias[slot_type])  ← 农民 / 雇工 / 市场三套选择偏置（仿 bias_vl）
                  weights = scores.gather(indices)，归一化后 × route_scale
路由专家 k 的输出 = SwiGLU 适配器(action_enc(desc(a_k)) + 专家 id 嵌入)   ← 仿 Expert，含 swiglu_limit
共享专家        = 现有神经路径 state_head(h)                               ← 仿 shared_experts
y = shared(h) + Σ_k w_k · E_k(a_k)
logits(a) = exp(s)·cos(y, z_a) + λ·Σ_k w_k·1[a = a_k]     （CLM 打分 + 指针复制，保证能精确复现某个技能）
→ 现有分层输出：操作 → 物品 → 数量
```
- **负载均衡**：按槽位类型分别做 noaux_tc 偏置更新（复用 `update_gate_bias` 的写法）。
- **分层配置**：参考 `get_moe_config`，农民、雇工、市场三类槽可以设置不同的专家数和 top-k。
- **安全初始化**：跟随计划技能的偏置设高，初始策略约等于最强启发式，之后靠学习在它之上改进（残差式策略学习）。
- **专家提议 token**：可选。每步把各技能的提议摘要编码成 token 放进序列，让注意力也能看到这些建议。

## 3. 关键步 RL（加速）
- **划分决策**：
  - **常规步**：移动、浇水、喂食、收集等。门控置信度高时，直接执行 top-1 技能，不采样，也不计入策略梯度。
  - **关键步**：市场槽（买什么种子和动物、雇几个人、买地、卖出时机和数量）、每天开始时的计划选择，以及门控熵高于阈值的不确定槽。
  - 这样每局需要随机采样和计入梯度的决策，从约 1 万次降到几百次，方差大幅下降。
- **关键步分叉 GRPO**：在关键决策点用 `FarmEnv.clone` 分出 G 个分支，每个分支采样不同动作，之后都用当前策略继续，优势 = 分支回报 − 分支均值。信用直接落在这一个决策上，比整局分组快得多。
  - 按 MiMo 公式 1：stop-grad ratio 乘以掩码，上下截断界解耦；按 CodeMidas，优势只减组均值，不做 std 归一化；加对中期策略的 KL。
  - 奖励：胜负 + 0.25·tanh(资金差/15000)，短分支用价值头 bootstrap。
- **对手**：公开启发式、自博弈快照（PFSP）、针对当前策略训练的 exploiter（动态对抗）。

## 4. 训练阶段（预训练 → 中期训练 → 后训练）
1. **预训练（已完成）**：`pre.pt`，作为 CSA2 主干、CLM 头和共享专家的初始化。
2. **中期训练**：
   - **replay 门控蒸馏**：在顶尖队伍的 replay 状态上（评分 ≥2950 的有 3.56 万局）跑出各技能的提议，以高手动作为目标训练门控和共享专家。门控由此学会"此刻哪个技能最像高手"。技能提议的生成放在 Kaggle CPU kernel 上并行。
   - **MOPD**：在策略自己访问到的状态上，只对关键步做短程 rollout，找出最优技能作为教师，蒸馏给策略。
3. **后训练**：关键步分叉 GRPO，GPU 负责学习，CPU worker 负责分支 rollout。

## 需要改动的关键文件
- **新建**：
  - `agent/skills.py`：技能库和适用性判断；
  - `model/hmoe.py`：HGate / HeuristicMoE；
  - `data/skill_labels.py`：在 replay 上生成技能提议；
  - `rl/keystep_grpo.py`：关键步分叉 GRPO。
- **修改**：
  - `model/clm_policy.py`：在打分之前插入 HMoE，并加指针复制；
  - `model/clm_batch.py`、`model/train_clm.py`：技能提议字段、关键步掩码、偏置更新；
  - `agent/clm_agent.py`：推理时运行技能、兜底执行 top-1 技能；
  - `rl/clm_rollout.py`：接入技能并支持分叉；
  - `submit/pack_clm.py`：打包技能代码。
- **复用**：
  - `model/dsv41.py` 的 Gate、Expert 写法和 `update_gate_bias`；
  - `env/fast_env.FarmEnv`（含 `clone`）、`data/replay_db.ReplayDB`、`rl/token_rollout.WorkerPool`；
  - `tools/eval_clm.py`、`tools/opening_probe.py`、`tools/bench_latency.py`、`tools/kaggle_gpu.py`。

## 日程
| 日期 | 工作 | 提交 |
|---|---|---|
| 09-25 | 写 `skills.py`（从 metav4 Chassis 拆出技能加通用技能）；影子模式验证；上限估计（每步选最优技能能多赢多少） | — |
| 09-26 | 写 `hmoe.py`（仿 DSV4.1）接入 `clm_policy`；单元测试（偏置只影响选择、强制门控能复现单个技能）；在 replay 上生成技能提议 | v3a：安全初始化版本 + 分层输出（≥ 最强启发式） |
| 09-27 | 中期训练：门控蒸馏 + MOPD（GPU ≤6 小时） | v3b |
| 09-28 | 关键步分叉 GRPO（联赛 + 自博弈 + exploiter） | v3c |
| 09-29 | 最终评估：≥100 局双座位；测延迟；最终提交 | v3d |

## 验证
- HGate 与 DSV4.1 语义一致：偏置只改变 indices，不改变 weights；归一化和 route_scale 数值与参考实现一致（单元测试）。
- 门控强制选技能 k 时，行为与技能 k 单独执行逐步一致；安全初始化版本本地不弱于最强启发式。
- 每个阶段：
  - 对每个联赛 agent 双座位 ≥50 局，统计胜率和资金；
  - 专家负载分布，检查是否塌缩到单个技能；
  - 关键步占比；
  - 延迟 p99 和最大值（2 线程，每步 1 秒限制）；
  - 不弃局测试。
