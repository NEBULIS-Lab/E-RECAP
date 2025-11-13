# Phase E: SDTP Ablation Study

**Note:** This document records the design of ablation experiments for future work, but is not in the current task list.

## E1. Ablation Purpose

SDTP's core innovations include:

1. Learnable token pruning module
2. Saliency-based supervision + ranking loss
3. Layer-wise dynamic pruning scheme
4. RoPE-compatible shrinking strategy
5. Head/tail token preserving strategy (前 4 + 后 16)
6. Multi-layer pruning schedule (10 layers)
7. Keep ratio 0.7, dynamic token length updates

The purpose of ablation studies is to prove that each component contributes to performance or speed, and removing any component will degrade the results.

---

## E2. Ablation Points by Category

**Legend:**
- **Performance**: Ablation points with significant impact on performance
- **Speed**: Ablation points with noticeable impact on speed
- **Analysis**: Ablation points for analyzing internal mechanisms (e.g., ranking, supervision methods)

---

### ① Pruning Capability (Core)

| Module | Ablation Method |
|--------|----------------|
| **Performance** Learnable Pruner | Replace with saliency-only baseline (remove MLP) |
| **Performance** Learnable Pruner | Replace with random pruning (keep pruning ratio unchanged) |
| **Performance** Layer-wise pruning | Replace with single-layer pruning (e.g., only prune layer 4) |
| **Performance** Layer-wise pruning | Replace with fixed layer set (e.g., fixed 3 layers) |
| **Speed** Keep ratio | Change from 0.7 to 0.8 / 0.5 / 0.3 |
| **Speed** Head/Tail Preserve | Remove "first 4 / last 16" preservation strategy |
| **Speed** RoPE Fix | Remove RoPE index fix → compare errors/performance degradation |

---

### ② Loss Function Related

| Module | Ablation Method |
|--------|----------------|
| **Analysis** Ranking Loss | Remove, keep only MSE + LM loss |
| **Analysis** MSE Loss | Remove saliency regression, use only ranking |
| **Analysis** LM loss | Remove LM loss (supervise only importance) |
| **Performance** Full supervision | Use only LM loss (no saliency) |

These experiments can verify the paper's claim that "ranking loss → improves ranking quality".

---

### ③ Inference Strategy Related

| Module | Ablation Method |
|--------|----------------|
| **Speed** Token selection | Use Top-k vs Top-p vs threshold-based |
| **Speed** Token merge | Do not update attention mask → observe error accumulation |
| **Speed** Multi-layer mask propagation | Use hard mask vs soft mask |

---

### ④ Multi-GPU / Communication Related (Focus on Speed)

| Module | Ablation Method |
|--------|----------------|
| **Speed** Multi-GPU pruning | Remove cross-GPU pre-pruning (speed decreases) |
| **Speed** Communication cost | Prune only in first 2 layers or last 2 layers (compare communication load changes) |

---

## E3. Ablation Experiment Matrix

**Table: SDTP Ablation Design Matrix**

| Ablation ID | Description | Variable Changed | Hypothesis | Expected Effect |
|-------------|-------------|------------------|------------|-----------------|
| A1 | Remove Pruner → Use saliency baseline | model=saliency-only | Learnable pruner performance > saliency | Performance ↓, Speed ≈ |
| A2 | Random pruning | importance=random | SDTP must learn ranking | Performance ↓↓ |
| A3 | No ranking loss | loss=LM+MSE | Ranking loss is important | Performance ↓ (moderate) |
| A4 | No MSE loss | loss=LM+Ranking | Regression alignment is critical | Performance ↓ (moderate) |
| A5 | Only LM loss (no saliency) | loss=LM | No supervision is weaker | Performance ↓ (significant) |
| A6 | Remove head/tail Preserve | no head/tail | Preservation strategy ensures semantic stability | Performance ↓ |
| A7 | Keep ratio 0.5 | ratio=0.5 | Over-pruning | Performance ↓, Speed ↑ |
| A8 | Keep ratio 0.3 | ratio=0.3 | Extreme pruning | Performance ↓↓, Speed ↑↑ |
| A9 | Keep ratio 0.8 | ratio=0.8 | Under-pruning | Performance ≈, Speed ↓ |
| B1 | Single-layer pruning | prune only L4 | Multi-layer is more effective | Performance ↓, Speed ↓ |
| B2 | No RoPE index fix | rope broken | Must fix | Model crashes |
| C1 | Soft mask only | no hard mask | Soft pruning accuracy higher or lower | Small change |
| C2 | No mask propagation | per-layer local | Upper layers will fail | Model instability |

You can combine 6–8 key ablation experiments for the final paper experiments.

---

## E4. Paper-Level Ablation Control Groups (Clear, Academic)

Each ablation experiment needs to compare:

### Baseline (Standard SDTP)

- Learnable pruning module (MLP)
- Loss: LM + MSE + Ranking
- Keep ratio = 0.7
- Head 4 + tail 16 preservation
- Layer-wise pruning (10 layers)
- RoPE index correction
- Attention mask and hidden state cascade updates

### Control Group (e.g., No-Ranking Loss)

**Ablation Settings:**
- Remove ranking loss
- Keep all other components unchanged

**Final Report Writing:**

"We isolate the contribution of the ranking loss by disabling it while keeping all other components fixed. This setting evaluates the effect of supervising the ordering of saliency scores."

---

## E5. Ablation Section (Paper Structure Template)

Below is a standard template based on top-tier conference writing style. You only need to fill in the results:

---

### 6. Ablation Studies

To better understand the contribution of each SDTP component, we conduct a comprehensive ablation study covering the pruning mechanism, supervision strategy, and inference-time pruning policy.

#### 6.1 Effect of the Learnable Pruner

(A1, A2 results)

#### 6.2 Contribution of Each Loss Component

(A3–A5 results)

#### 6.3 Influence of Pruning Ratio

(A7–A9 results)

#### 6.4 Role of Head/Tail Preservation

(A6 results)

#### 6.5 RoPE Index Correction Is Essential

(B2 results)

#### 6.6 Multi-Layer vs Single-Layer Pruning

(B1 results)

#### 6.7 Communication Reduction in Multi-GPU Inference

(C1–C2 + multi-GPU results)

---

# Phase E: SDTP 消融实验设计（中文版）

**说明：** 本文档记录后续消融实验的设计，但不在当前任务列表中。

## E1. 消融实验目的

SDTP 的核心创新点包括：

1. 可学习 token 剪枝模块（Learnable token pruning module）
2. 基于显著性的监督 + 排序损失（Saliency-based supervision + ranking loss）
3. 分层动态剪枝策略（Layer-wise dynamic pruning scheme）
4. RoPE 兼容的收缩策略（RoPE-compatible shrinking strategy）
5. 头尾 token 保留策略（前 4 + 后 16）
6. 多层剪枝调度（10 层）
7. 保留率 0.7，动态 token 长度更新

消融实验的目的是证明每个组件都能贡献性能或速度，如果移除会使效果下降。

---

## E2. 按模块分类的消融点

**图例：**
- **性能影响**：对性能影响最大的消融点
- **速度影响**：对速度影响明显的消融点
- **机制分析**：用于分析机制内部的消融点（如排序、监督方式）

---

### ① 剪枝能力相关（核心）

| 模块 | 消融方式 |
|------|---------|
| **性能影响** 可学习剪枝器 | 用 saliency × MLP 改为纯 saliency 基线 |
| **性能影响** 可学习剪枝器 | 改为随机剪枝（保持剪枝率不变） |
| **性能影响** 分层剪枝 | 改为单层剪枝（例如只剪第 4 层） |
| **性能影响** 分层剪枝 | 改为固定层集合（例如固定 3 层） |
| **速度影响** 保留率 | 0.7 → 0.8 / 0.5 / 0.3 |
| **速度影响** 头尾保留 | 移除"前 4 / 后 16"保留策略 |
| **速度影响** RoPE 修正 | 移除 RoPE index fix → 对比报错/性能下降 |

---

### ② 损失函数相关

| 模块 | 消融方式 |
|------|---------|
| **机制分析** 排序损失 | 移除，只保留 MSE + LM loss |
| **机制分析** MSE 损失 | 移除 saliency 回归，仅用排序 |
| **机制分析** LM 损失 | 移除 LM loss（仅监督重要性） |
| **性能影响** 全监督 | 仅使用 LM loss（无 saliency） |

这些实验可以验证论文声称的"排序损失 → 提升排序质量"。

---

### ③ 推理阶段策略相关

| 模块 | 消融方式 |
|------|---------|
| **速度影响** Token 选择 | 使用 Top-k vs Top-p vs 基于阈值 |
| **速度影响** Token 合并 | 不更新 attention mask → 观察错误累计 |
| **速度影响** 多层 mask 传播 | 使用硬 mask vs 软 mask |

---

### ④ 多 GPU / 通信相关（主要看速度）

| 模块 | 消融方式 |
|------|---------|
| **速度影响** 多 GPU 剪枝 | 移除跨卡前剪枝（速度下降） |
| **速度影响** 通信成本 | 剪枝仅在前 2 层或后 2 层（对比通信负载变化） |

---

## E3. 消融实验矩阵

**表：SDTP 消融设计矩阵**

| 消融 ID | 描述 | 变量改变 | 假设 | 预期效果 |
|---------|------|----------|------|----------|
| A1 | 移除剪枝器 → 使用 saliency 基线 | model=saliency-only | 可学习器性能 > saliency | 性能 ↓，速度 ≈ |
| A2 | 随机剪枝 | importance=random | SDTP 必须学习排序 | 性能大幅 ↓ |
| A3 | 无排序损失 | loss=LM+MSE | 排序损失很重要 | 性能 ↓（中度） |
| A4 | 无 MSE 损失 | loss=LM+Ranking | 回归对齐很关键 | 性能 ↓（中等） |
| A5 | 仅 LM 损失（无 saliency） | loss=LM | 无监督更弱 | 性能 ↓（明显） |
| A6 | 移除头尾保留 | no head/tail | 保留策略保证语义稳定 | 性能 ↓ |
| A7 | 保留率 0.5 | ratio=0.5 | 剪枝过度 | 性能 ↓，速度 ↑ |
| A8 | 保留率 0.3 | ratio=0.3 | 极度剪枝 | 性能大幅 ↓，速度 ↑↑ |
| A9 | 保留率 0.8 | ratio=0.8 | 剪枝不足 | 性能 ≈，速度 ↓ |
| B1 | 单层剪枝 | prune only L4 | 多层更有效 | 性能 ↓，速度 ↓ |
| B2 | 无 RoPE index 修正 | rope broken | 必须修正 | 模型崩溃 |
| C1 | 仅软 mask | no hard mask | 软剪枝准确率更高或更低 | 变化小 |
| C2 | 无 mask 传播 | per-layer local | 上层会失败 | 模型不稳定 |

可以通过 6–8 个重点消融实验组合成最终论文实验。

---

## E4. 论文级别的消融对照组（清晰、学术）

每个消融实验都需要对比：

### 基线（标准 SDTP）

- 可学习剪枝模块（MLP）
- 损失：LM + MSE + Ranking
- 保留率 = 0.7
- 头部 4 + 尾部 16 保留
- 分层剪枝（10 层）
- RoPE index 修正
- Attention mask 和 hidden state 级联更新

### 对照组（例如：无排序损失）

**消融设置：**
- 移除排序损失
- 其他保持一致

**最终报告写作：**

"我们通过禁用排序损失来隔离其贡献，同时保持所有其他组件不变。此设置评估了监督 saliency 分数排序的效果。"

---

## E5. 消融章节（论文结构模板）

以下是基于顶会写作风格的标准模板，您只需要填入结果：

---

### 6. 消融研究

为了更好地理解每个 SDTP 组件的贡献，我们进行了全面的消融研究，涵盖剪枝机制、监督策略和推理时剪枝策略。

#### 6.1 可学习剪枝器的效果

(A1, A2 结果)

#### 6.2 每个损失组件的贡献

(A3–A5 结果)

#### 6.3 剪枝率的影响

(A7–A9 结果)

#### 6.4 头尾保留的作用

(A6 结果)

#### 6.5 RoPE Index 修正是必需的

(B2 结果)

#### 6.6 多层 vs 单层剪枝

(B1 结果)

#### 6.7 多 GPU 推理中的通信减少

(C1–C2 + 多 GPU 结果)
