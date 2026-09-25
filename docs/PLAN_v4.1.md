# Kaggriculture v4.1：cha22 骨架 × 可学习插入点 × Engram 对手记忆（替换 TTT）

## Context
- **当前方案**：v4（`docs/PLAN_v4.md`，已推送到 edd6901）。
  - cha22 作为程序骨架，只负责单位动作和路线动作带。
  - 在它手写、手调的决策点上插入学习器：H1 卖出、H2 肥料、H3 原料采购、H4 雇工、H5 路线、H6 对手识别。
  - 先离线 RL，再做关键步 GRPO。
  - 对手模型原本用 TTT。
- **这次的改动**：用户要求把 TTT 换成 Engram。Engram 是 DeepSeek 的"条件记忆"：对 N-gram 做哈希、O(1) 查表，再用上下文门控写入残差流。实现照搬官方 DeepSeek-V4.1-Flash 的 `inference/engram.py` 和 `model.py`。
- **为什么 Engram 适合这个比赛**（本次会话实测）：
  1. **天梯对手高度重复**：最近 6,000 局、12,000 个玩家局里，"前两天下单序列"只有 1,922 种。前 10 种覆盖 23.9%，前 58 种覆盖 50%，前 232 种覆盖 75%；每支队伍的签名数中位数是 1。大多数对手是确定性程序，查表记忆最擅长的正是这类重复、固定的模式。
  2. **cha22 本身就在做手写的查表识别**：CTRTABLE 在第 2 步用"对手资金 + 市场小麦库存"识别已知对手的动作带。Engram 是它的通用、可学习版本。
  3. **Engram 要哈希的 token 在正式比赛中拿得到**：只凭观测就能 99.87% 精确地推算出对手每步的市场流量（`tools/opp_flow_check.py`），对手的农场变化本来就是公开的。
- **G1 进展**（H1 卖出插入点的上限，完美预知卖出器 vs cha22，种子 9401）：
  - 不加现金保护：−67k。原因是囤货导致动作带里的买动物失败（8 头 vs 16 头）。
  - 加现金保护（现金低于 3,000 时照用 cha22 的卖单）：−4.3k。
  - 剩下的亏损来自仓库溢出：仓库装满的天数是 7 天，原版只有 2 天。外生库存路径的预测在大多数商品上完全准确。
  - H5 还没跑。

## 照搬的官方 Engram（DeepSeek-V4.1-Flash）
- **`NgramHashState`**：
  - 先把 token 映射到压缩词表。
  - 每个位置取最近 `max_ngram_size` 个压缩 token；序列开头和 DEAD 段用 pad 填充，保证 N-gram 不会跨越它们。
  - 每个"(层, 回看位置)"配一个奇数乘子，由 `default_rng(10007·layer_id)` 生成，上界保证乘积不会溢出 int64。
  - 沿回看方向做滚动 XOR，依次得到 2 到 N 阶 N-gram 的哈希。
  - 每个"(阶数, 头)"对一个专属素数取模。素数从 `engram_vocab_size` 起往上找，且不重复使用；再加上偏移，使各段在同一张表里互不重叠。
- **`Engram` 模块**：
  - 取出 (N−1)·heads 行，展平后经过 `wkv`，得到每个 hc 副本各一个 key，外加一个共享的 value。
  - 门控 = sigmoid(带符号开方(RMS 规范化后的 (h·q_w·k_w)·key / √dim))，输出 `h + gate·value`。
  - `token_mask` 可以把门关掉。
  - 在 `Block` 之前调用：`h = layer.engram(h, hashes)`。
- **Flash 的配置**：`engram_layer_ids=[1,14]`（共 40 层）、`max_ngram_size=4`、`n_heads=8`、`head_dim=256`、素数从 1600 万起找、每层约 3.84 亿行（fp8 + 分块 scale）。

## 用户决定：Engram 挂在预训练的 CSA2（pre.pt）上
- **pre.pt 的结构**（`scratchpad/gpu_outputs/evelynyang02__kgc-train-pre/ckpt/`，文件 18 MB）：dim=128，6 层，hc_mult=2（带 hyper-connection，与 Flash 同构），FFN 为 MoE（4 路由 + 1 共享），窗口 128，CSA2 压缩注意力。每步输入 33 个观测 token + `<ACT>`，推理时用增量 KV 缓存（`agent/clm_agent.py`）。
- **插入位置**：照 Flash 的相对深度（第 1、14 层 / 共 40 层），在 Block 1 之前放 S 流 Engram，在 Block 3 之前放 D 流 Engram，写法与官方一致：`h = engram(h, hash_ids, token_mask)`。
  - 模块完全照搬官方代码：每个 hc 副本一个 key，外加一个共享 value；门控用带符号开方后的 sigmoid。
  - `wkv` 中 value 部分的权重初始化为 0，这样起步时网络与 pre.pt 完全一致，再开始微调。
- **哈希 id 怎么对齐到位置**：第 t 步的全部 token（33 个观测 token + `<ACT>`）共用同一组哈希 id，即 S 流上以第 t 步结尾的 2–4 阶 N-gram，加上 D 流上以当天结尾的 2–4 阶 N-gram。门控按每个位置的隐状态分别计算。
  - 原因：观测 token 的内容在连续特征里，token id 本身没有语义，所以不像文本那样直接对输入 id 做 N-gram，而是改用对手事件流。
- **新增输出头：CLM 式对手事件预测**。用 `<ACT>` 位置合并 hc 后的隐状态算出查询向量，与"候选对手事件"的编码算 exp(logit_scale)·cos，用 InfoNCE 训练。
  - 候选事件由描述特征编码：商品、买或卖、数量（数值特征）、预测跨度（1/4/24/72 步、到终局）。
  - 候选集合就是 S 流的压缩事件词表，与 Engram 共享同一个事件空间。
  - 复用 `model/clm_policy.py` 的候选编码器和余弦打分；InfoNCE 的实现和原 CLM 头相同。
  - 倾销时刻、隐藏库存用小回归头预测；另外输出对手表示 z_t。
  - 原来预测**我方**动作的 CLM 头在 v4 里只作为辅助损失保留，不再用来出动作。
  - 后续插入点的学习器（H1、H3、H5 的小头）也读这个隐状态。
- **训练**：
  - 从 pre.pt 开始，在已有的 542 个 CLM 分片上微调，并按 (episode_id, seat) 拼接新抽取的对手事件数据。
  - 损失 = 对手预测 NLL（主）+ 原 CLM 策略损失 × 0.3（辅助，防止表示遗忘）。
  - 在 GPU notebook 上训练，约 3–4 小时，沿用 `model/train_clm.py` 的 DDP、同步停止和按时间计算的学习率。
- **推理**：v4 agent = cha22 骨架 + 插桩的插入点。CSA2+Engram 每步只喂"观测 chunk + `<ACT>`"，不再逐槽位解码动作，输出给各插入点用。延迟目标：p99 < 300 ms（2 线程）。

## 我们的版本：缩小后的"对局 Engram"
- **两条 token 流**。压缩就是离散化：
  - **S 流（每步一个 token）**：由三部分组成：小时段；对手 9 种商品的净流量，各自分桶为 {0, 1–2, 3–5, 6–10, 11+}；对手可见农场的变化类型（种了什么、买了什么动物、雇了几人、买了地）。
  - **D 流（每天一个 token）**：对手当天各商品的卖出和各类买入，分桶后压缩成一个 token。此外在第 2 步额外放一个"指纹 token"，即"对手资金 + 市场小麦库存"，也就是 cha22 CTRTABLE 用的键。
- **两个 Engram 层，照 Flash 分"浅层 + 中层"**：
  - 浅层哈希 S 流的 2–4 步 N-gram，记住短期模式；
  - 中层哈希 D 流的 2–4 天 N-gram（包括指纹），用来识别对手是哪套程序、处在哪个阶段。
  - 参数：每层 4 个头，每行 32 维，素数从 2^15 起找，每层 12 段共约 39 万行；用 int8 加分块 scale 存储，每层约 12 MB。
- **主干**：预训练的 CSA2（见上一节）。按官方写法，Engram 用主干的隐状态做门控：记忆和当前上下文匹配时，门才打开。
- **token 压缩词表**：从训练数据中统计 S、D 两种事件元组，按频率建立压缩词表，罕见元组归入 UNK，相当于官方 `build_compressed_token_map` 的作用。哈希乘子的上界由压缩词表的大小决定，与官方一致。
- **输出头**：
  - 对手各商品未来 1 / 4 / 24 / 72 步以及到终局的事件（CLM 式候选打分，见上）；
  - 对手的倾销时刻；
  - 对手的隐藏库存；
  - 对手表示 z_t，作为特征提供给 H1（卖出）、H3（原料采购）、H5（路线）、H6（对手识别）。
- **训练（离线）**：
  - 数据：用官方解释器精确重放 73,406 局 replay，两个座位都用，标签是另一方实际成交的流量。
  - 哈希表通过反向传播学习（稀疏 Adam）。
  - 数据按队伍划分：90% 的队伍用于训练。评估时分开两类：没见过的队伍，以及见过的队伍的留出对局。
- **怎么更新**：
  - 测试时记忆是静态的，只做 O(1) 查表，没有梯度更新。但每步的上下文在变，查到的记忆也随之变化。
  - 跨局更新：抓回新的天梯 replay，重新训练，或直接为新出现的 N-gram 写入新行，然后重新提交。

## 用官方 Engram 替换 cha22 的查表（落实 H5、H6）
cha22 里有三处手写查表，机制与 Engram 相同：键是写死的签名，值是作者手调的决定。
| cha22 的表 | 现在的键 → 值 | 换成 Engram |
|---|---|---|
| 路由表 `_R108_SHOP_ROUTES` / `_R110_OLD_SHOPS` / `_V92_TABLE`（第 144 步） | 前两个商店 → 路线编号（41 条之一） | N-gram 键：商店 + 对手指纹 + 对手前几天的 D 流；值为学到的 41 条路线打分 |
| CTRTABLE（第 2 步） | 对手资金 + 小麦库存 → 针对已知对手的小麦对冲 | 同一指纹 + S 流 N-gram；值为学到的对冲决定，覆盖表外对手 |
| 路由器的 `rkey`（第 2 步） | 对手签名 → 供后续补丁使用 | 由 Engram 输出的对手表示 z_t 取代 |
- 模块：复用 `model/engram.py`（与官方逐位一致），记忆写入 CSA2 当步隐状态，再接小头输出路线打分或对冲决定。
- 安全初始化：打分 = cha22 原表的选择（大的 one-hot 偏置）+ Engram 贡献；value 初始为 0，起步与 cha22 完全一致。
- 训练：路线用 H5 上帝视角分叉的真实终局收益（离线 RL → GRPO）；对冲用 replay 中对手的真实早期行为。

## RL 阶段的数据配比（09-25 夜间定稿）
目标是天梯排名，所以数据分布要贴近"我们实际会遇到的对手"，并重点修正我们输掉的局面。

| 类别 | 来源 | 离线 RL（H5 路线数据 / IQL） | 在线 GRPO | 理由 |
|---|---|---|---|---|
| loss：我们输掉的天梯对局 | 爬虫抓回 v4a 等提交的对局；按原种子开环重放当局对手的动作 | **30%** | 20% | 最直接的"错题本"；市场是唯一交互渠道，开环重放几乎就是真实对手 |
| near：分数相近的天梯对手（2200–2600） | replay 库中该分数段的对局，对手动作开环重放 | **30%** | 20% | 天梯上的对局主要就来自这个分数段 |
| top：2700+ 顶尖队伍 | replay 开环重放 | 15% | 10% | 往上爬必须能对付的对手 |
| live：联赛 agent（会随价格反应） | cha22、metav4、v48、farm2945 + 6 个公开 agent | 15% | 20% | 开环重放不会反应，需要会反应的对手补足博弈部分 |
| self：自博弈 | 当前策略及历史快照（PFSP，按胜率加权偏向难对手） | 10% | **30%** | 在线阶段防止被针对、学抢卖博弈；离线阶段只需少量 |

- 实现：`tools/route_data.py` 的 `ROUTE_MIX="loss=0.3,near=0.3,top=0.15,live=0.25"`（live 里包含 cha22 镜像，即离线阶段的自博弈部分）。loss / near 依赖本地爬虫实时更新的 replay 库，所以在本地跑；Kaggle kernel 跑 live + top。
- 爬虫以 v4a（submission 56553173）为种子持续采集，之后每个新提交都加入 `OUR_SUBS`，新输掉的局自动进入 loss 池。
- CSA2+Engram 的对手记忆训练用全部 replay（两个座位），不做配比：记忆需要覆盖尽可能多的对手程序。

## 工作顺序
1. **收尾 G1**：
   - 收紧 `tools/hook_oracle.py` 的仓库保护：白天最多囤 60 件；日终预测"仓库 + 携带物品"时留出单位拾取的余量，控制在 ≤ 90 件。
   - 跑 H1：10 个种子 × 2 个座位，分开环和闭环。
   - 跑 H5：6 个种子 × 2 个座位。
   - 按上限给插入点排序。
2. **准备数据**：只重放环境，不重建观测 token，生成对手事件数据（每步双方实际成交的流量、可见的农场变化、第 2 步指纹、商店）。
   - 新建 `agent/opp_events.py`：从相邻两步观测和我方动作推算对手事件。逻辑来自 `tools/opp_flow_check.py`，实测 99.87% 精确；训练和正式对局共用这一份代码，保证两边一致。
   - 新建 `data/opp_flow_extract.py`：复用 `data/replay_db.ReplayDB.rebuild`；本地 4 核跑完 73k 局约需 1–1.5 小时。输出按 (episode_id, seat) 索引，与已有 CLM 分片拼接。
3. **Engram 模型与训练**：
   - `model/engram.py`：移植官方的 `EngramLayout`、`NgramHashState`、`Engram`，改成 int8 表，hc_mult=2。
   - `model/dsv41.py` 和 `model/clm_policy.py`：在 Transformer 的层循环里，于 Block 1 和 Block 3 之前调用 Engram；新增对手预测头和 z_t 输出。
   - `model/clm_batch.py` 和 `model/train_clm.py`：增加 S/D 哈希 id 和对手流量标签字段，新增 `--init pre.pt`、`--engram` 等参数；通过 `tools/kaggle_gpu.py` 在 GPU notebook 上训练。
4. **插桩**：新建 `agent/hooks.py`，包装 cha22 的决策函数。安全初始化：学习器 = cha22 的原决定 + 初始为 0 的修正量，要求逐位一致。在上限排名靠前的插入点装学习器，输入包括 Engram 给出的特征。
5. **训练学习器**：先离线 RL（在分叉数据上做 IQL/AWR），再做关键步分叉 GRPO。
6. **清理与提交**：
   - 删除 MoE 代码：`agent/expert_pool.py`、`model/hmoe.py`、`agent/clm_agent.py` 里的 pool 路径。
   - 把 `docs/PLAN_v4.md` 中的 TTT 全部改成 Engram。
   - 修改 `submit/pack_clm.py`，打包 Engram 权重。
   - 测延迟、评估、提交。

## 关键文件
- **新建**：`model/engram.py`、`agent/opp_events.py`、`data/opp_flow_extract.py`、`agent/hooks.py`、`tools/test_engram.py`。
- **修改**：
  - `model/dsv41.py`、`model/clm_policy.py`：插入 Engram 层，新增对手预测头；
  - `model/clm_batch.py`、`model/train_clm.py`：新增哈希 id、标签字段和从 pre.pt 初始化；
  - `agent/clm_agent.py`：只喂观测 chunk，删除 pool 路径；
  - `submit/pack_clm.py`：打包 Engram 表；
  - `tools/hook_oracle.py`：收紧仓库保护；
  - `docs/PLAN_v4.md`：TTT 全部改成 Engram。
- **复用**：`tools/opp_flow_check.py`、`infra/fork.py`（`fork_branches`、`warm_pool`）、`env/fast_env.py`、`data/replay_db.py`、`data/seq_extract.py` 的分片格式、`tools/kaggle_gpu.py`、`tools/bench_latency.py`。

## 验证
- **`tools/test_engram.py`**：
  - 在玩具词表上，哈希 id 与官方算法逐位一致：滚动 XOR、各素数段不重叠、序列开头用 pad。
  - `token_mask` 能关掉门。
  - 查表结果是确定的。
- **G2**：在留出对局上比较对手事件预测的 InfoNCE 损失 / top-1 准确率，以及换算成的流量 MAE，主干 vs 主干 + Engram，并分"见过的队伍 / 没见过的队伍"两组报告。预期见过的队伍大幅改善，没见过的队伍不变差。
- **安全初始化**：Engram 的 value 权重初始化为 0 时，插入 Engram 后的 CSA2 输出与 pre.pt 逐位一致。
- **延迟和包大小**：用 `tools/bench_latency.py` 在 2 线程 CPU 上测，每步"cha22 + CSA2 观测 chunk + Engram" p99 < 300 ms；提交包大小合规（pre.pt 18 MB + 两张 int8 表约 25 MB）。
- **端到端**：v4 agent 双座位 ≥ 200 局，对手为 cha22 和联赛；vs cha22 胜率 > 55%。

## 日程（截止 09-30）
| 日期 | 工作 | 提交 |
|---|---|---|
| 09-25 | 收尾 G1；开始重放 replay | — |
| 09-26 | Engram 模型、训练、G2；插桩 | — |
| 09-27 | 学习器 + 离线 RL | v4a / v4b |
| 09-28 | GRPO | 最终候选 |
| 09-29 – 30 | 监控、热修复 | — |
