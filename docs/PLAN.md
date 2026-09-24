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
  3. **不要分类头**，模型直接**解码输出动作 token**。
  4. 预训练可以用 GPU，但必须保证 GPU 利用率高，不能卡在 CPU 数据加载上。RL 也可以用 GPU。其他情况非必要不用 GPU。

## 关键发现（影响设计）
- 如果把基线的卖出量压成粗分档，每局会少约 1.2 万资金。所以动作必须**精确表达**，这也支持用 token 解码。
- top 对局接近镜像，胜方和负方的资金只差约 2.5%。top 选手经常分批卖出（1 个、25%、50%）。
- metav4 打 metav4 是确定性平局；手写的卖出偏离规则都会输。
- **open-loop 回放 top 选手的动作不能当对手用**：双方共享随机数流（杂草生成、商店解锁），一方换人，对方的局面就变了。所以 replay 只用于蒸馏训练和分析，评估必须用闭环 agent。

## 模型：编码器 + TTT 时间记忆 + 自回归动作解码器
- **动作 token 化**（`agent/action_tokens.py`）：每步的完整动作编码成一串 token。
  - 例：`<FARMER> PLANT WHEAT <HAND> NORTH … <MARKET> SELL MELON 7 BUY_PRODUCT WHEAT 20 <EOS>`
  - 词表：单位操作、物品、数量（0–99 精确值，外加 ≥100 的分档和 999 表示全部）、分隔符、`<BASE>`（照抄执行层动作的兜底 token）。
  - 要求在 replay 上**编码→解码后 100% 还原原始动作**，以此作为验收。
- **编码器（每步一次）**：实体 Transformer。
  - 输入 token：双方 200 个地块（带 2D 位置和本方/对方编码）、9 个商品、全局、单位位置与背包。
  - d=128 左右，2–3 层。
- **时间主干**：24 步滑窗注意力（1 天），加 In-Place TTT 快速权重层。
  - 快速权重每天结束时用闭式梯度更新一次。
  - 自监督目标：当天各商品的市场净流量和价格变化，都可以在对局中直接观测到。
  - 初始快速权重通过元学习得到。
- **解码器**：小型因果 Transformer，对编码器和时间主干的输出做交叉注意力，自回归生成动作 token。
  - 解码时按语法掩码，只允许合法 token，例如 SELL 后只能接物品再接数量，移动或种植必须合法。
  - 推理用 KV 缓存，每步约 70 个 token，numpy 实现，目标每步 <100ms。
- **价值头**：保留，预测终局胜率和资金差，供 RL 做信用分配。
- **numpy 推理版与 torch 训练版**必须逐位对拍一致，沿用现有 `tools/test_equivalence.py` 的做法。

## 训练三阶段
1. **预训练（GPU，全量 replay 蒸馏）**
   - 数据：评分 ≥1800 的全部对局，双方视角。
   - 损失：
     - 下一个 token 的交叉熵，按评分和胜负加权（胜方 1.0、负方 0.5）
     - 价值头损失
     - TTT 自监督损失
   - 大 batch、混合精度（AMP）、按块预取。
2. **中期训练（GPU，高质量蒸馏）**
   - 数据：评分 ≥2800、以胜方为主的子集，较低学习率。
   - 做法：继续蒸馏；TTT 在大量不同对手的真实对局上继续元学习。
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
  - 使用 AMP 混合精度；TTT 按天做向量化 reshape，不再用 nonzero 索引。
  - 每轮打印 GPU 利用率和吞吐（token/s）作为监控。

## 数据
- replay 采集会继续直到全部下完（约 4 万多局）。
- Kaggle 上的公开 Notebook 和数据集会一起更新，采集程序支持断点续跑。

## 仓库结构（分支 `claude/kaggriculture-rl-solution-ayrf1p`）
```
env/            fast_env.py, replay_check.py
data/           crawl.py, replay_db.py, extract.py(→ 增加 token 序列), pack.py
agent/          features.py, action_tokens.py(新), controller.py, main 打包
model/          net.py(→ 编码器+TTT+解码器), policy_np.py(numpy 解码器), pretrain.py(GPU 预训练/中期训练)
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
