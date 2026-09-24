# Kaggriculture：Replay 蒸馏 + 强化学习方案（v2，按最新要求更新）

## Context
- **比赛**：两人对战农场模拟，终局钱多者胜。
  - 排行榜是动态评级，榜首约 3150–3290。
  - 截止 2026-09-30。
  - 每步 `actTimeout=1s`，推理只能用 CPU。
- **已完成**：
  - 本地环境直接驱动官方 interpreter，已与线上 replay 逐步对拍，720 步完全一致。
  - 可断点续跑的大规模 replay 采集：已发现约 4.4 万局，下载中。
    - 公开数据集 `xishengfeng/kaggriculture-replay-db`
    - 公开 Notebook `xishengfeng/kaggriculture-replay-database`
  - 基线提交 metav4，线上评级收敛中。
- **用户的最新要求**：
  1. 训练严格按 **预训练 → 中期训练 → 后训练（RL）** 的顺序进行。RL 必须在大规模 replay 蒸馏之后才开始，之前提前启动的 GRPO 已停止。
  2. 预训练和中期训练**只用大规模 replay 做蒸馏**，不用手写规则对手，也不用我们自己执行层的对局数据。
  3. **不要分类头**，采用 **decoder-only** 统一序列模型（参考 DeepSeek-V4.1-Flash 的 CSA2 稀疏注意力），直接解码输出动作 token，不设独立编码器。
  4. 预训练可以用 GPU，但必须保证 GPU 利用率高，不能卡在 CPU 数据加载上。RL 也可以用 GPU。其他情况非必要不用 GPU。

## 关键发现（影响设计）
- 如果把基线的卖出量压成粗分档，每局会少约 1.2 万资金。所以动作必须**精确表达**，这也支持用 token 解码。
- top 对局接近镜像，胜方和负方的资金只差约 2.5%。top 选手经常分批卖出（1 个、25%、50%）。
- metav4 打 metav4 是确定性平局；手写的卖出偏离规则都会输。
- **open-loop 回放 top 选手的动作不能当对手用**：双方共享随机数流（杂草生成、商店解锁），一方换人，对方的局面就变了。所以 replay 只用于蒸馏训练和分析，评估必须用闭环 agent。

## 模型：Decoder-only 统一序列（参考 DeepSeek，无独立编码器）
- **一局 = 一条 token 序列**：每步依次是 `[观测 token][动作 token]`。一局约 720 × ~70 ≈ 5 万 token。
- **观测 token（每步约 30 个）**：连续特征经线性投影直接得到 token 嵌入，类似 VLM 的 patch embedding，不再使用编码器堆栈。
  - 9 个商品 token、1 个全局 token
  - 双方各 4 个象限 token（每个象限 25 格特征展平后投影）
  - 单位 token（农民和雇工的位置与背包）
  - 另加类型嵌入和步内位置嵌入
- **动作 token**（`agent/action_tokens.py`）：离散词表，包括单位操作、物品、数量（0–99 精确值，≥100 的分档，999 表示全部）、分隔符 `<FARMER>/<HAND>/<MARKET>/<EOS>`，以及兜底 token `<BASE>`。
  - 验收要求：在 replay 上编码→解码 100% 还原原始动作。
- **主干：参照 DeepSeek-V4.1-Flash 的 CSA2（已读官方 inference/model.py 和 config）**
  - 每一层都同时看两部分 KV，合并成一次注意力：
    - 最近 128 个 token 的原始 KV（滑窗，约 2 步）；
    - 由轻量索引器从整局历史中选出的 top-k 个压缩位置（k≈64–128）。这是 DSA / lightning indexer 的做法。
  - 各层的 `compress_ratios`（参考 V4.1 的 0 / 2 / 1）：第 0 层只用滑窗；中间层压缩比为 2（每 2 个 token 用门控 softmax 池化成 1 个 KV）；后几层压缩比为 1（逐 token 做 top-k）。
  - KV 和索引跨层复用（Full / Reindex / Reuse）：只有少数 kv_source 层做压缩，少数 index_source 层跑索引器，其余层直接复用。
  - 不用稠密全注意力。原因是 CPU 推理每步只有 1 秒，而一局约 5 万 token；CSA 能让每个 token 的计算量有上限，同时保留全局视野。
  - MTP（`n_mtp_layers`=1–2）：训练时作为辅助损失，推理时可做投机解码。
  - MoE、mHC、Engram、CED 在当前规模（约 1M 参数）下不引入。
  - 价值头：预测终局胜率和资金差，供 RL 使用。
- **损失**：只在动作 token 上计算 next-token 交叉熵（蒸馏），另加价值头和 MTP 辅助损失。
- **推理**：numpy 实现，带 KV 缓存。观测 token 一次性并行 prefill，动作 token 逐个解码，按语法掩码只生成合法动作。目标每步 <100ms。
- numpy 推理版与 torch 训练版逐位对拍一致。

## 训练三阶段
1. **预训练（GPU，全量 replay 蒸馏）**
   - 数据：评分 ≥1800 的全部对局，双方视角。
   - 损失：
     - 下一个 token 的交叉熵，按评分和胜负加权（胜方 1.0、负方 0.5）
     - 价值头损失
     - MTP 辅助损失
   - 大 batch、混合精度（AMP）、按块预取。
2. **中期训练（GPU，高质量蒸馏）**
   - 数据：评分 ≥2800、以胜方为主的子集，较低学习率。
   - 做法：继续蒸馏。
   - 验收：蒸馏出的 agent 在本地闭环对局中与 metav4 和其他公开 agent 对打，看胜率与资金。
   - 若弱于 metav4，则启用 `<BASE>` 混合模式：在执行不稳定的部分照抄执行层动作。
3. **后训练（RL，GPU 负责学习，CPU 负责 rollout）**
   - 从中期训练的权重初始化，做 GRPO 自博弈联赛，在 token 级别计算 PPO 截断：
     - 对手：自身快照（PFSP）、Main/League Exploiter，以及公开 agent 作为固定对手。规则对手只允许在 RL 阶段使用。
     - 奖励：胜负加 GAR 式的资金差奖励；丢弃全胜或全负的组（Dynamic Sampler）；组内相对优势。
     - 价值头提供逐步优势；加一个锚定到中期训练模型的 KL 项。

## GPU 数据管线（保证利用率）
- CPU 端只做一次重活：用 `data/extract.py` 按分片增量重放并抽取特征，再用 `data/pack.py` 打包成定长 720 步、float16/int8 的连续分块，每块约 2048 条轨迹。
- 抽取和打包在 CPU 上完成（本地加 Kaggle CPU Notebook 分片并行），产物作为数据集或 Notebook 输出交给 GPU Notebook。
- GPU 训练端：
  - 后台线程把下一块 mmap 进来，放入 pinned 内存，再异步拷到显存。
  - 训练时直接在显存里随机取小批次，训练循环里没有 CPU 解压、拼 batch 或 Python 逐条处理。
  - 使用 AMP 混合精度；训练时稀疏注意力用块化的掩码实现，保证与推理时的 top-k 选择语义一致。
  - 每轮打印 GPU 利用率和吞吐（token/s）作为监控。

## 数据
- replay 采集会继续直到全部下完（约 4 万多局）。
- Kaggle 上的公开 Notebook 和数据集会一起更新，采集程序支持断点续跑。

## 仓库结构（分支 `claude/kaggriculture-rl-solution-ayrf1p`）
```
env/            fast_env.py, replay_check.py
data/           crawl.py, replay_db.py, extract.py(→ 增加 token 序列), pack.py
agent/          features.py, action_tokens.py(新), controller.py, main 打包
model/          net.py(→ decoder-only, CSA2 风格: 滑窗+压缩KV+top-k 索引器+MTP), policy_np.py(numpy KV-cache 解码), pretrain.py(GPU 预训练/中期训练)
rl/             grpo.py(后训练, token 级), rollout.py, league.py, rule_adversaries.py(仅 RL 阶段)
notebooks/      replay_db / extract(CPU) / pretrain(GPU) / rl(GPU) 的 Kaggle kernel 构建器
submit/         pack.py
```

## Verification
- `replay_check`：随机抽 top replay，逐步完全一致。
- 动作 token 化：随机抽 1000 局，编码→解码与原动作完全一致。
- numpy 与 torch 输出对拍，误差 <1e-5。
- 预训练和中期训练：验证集上的下一个 token 准确率（整体和"非 PASS"部分分开统计）、价值头损失，以及 GPU 利用率日志。
- 蒸馏 agent：本地闭环双座位 ≥50 局，对 metav4 和其他公开 agent 统计胜率；单步耗时 <300ms，零报错。
- RL：留出 seed 上的胜率曲线，以及熵、截断比例、全胜/全负组的比例。
- 提交前：双座位 ≥100 局，对全部公开 agent 胜出，零报错。
