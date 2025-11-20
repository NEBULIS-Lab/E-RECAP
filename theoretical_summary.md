# SDTP论文：数学与理论总结

## 一、核心问题与动机

### 1.1 计算复杂度问题
- **问题**：注意力机制具有二次计算复杂度 $O(N^2)$，其中 $N$ 为序列长度
- **影响**：当序列长度从 4K 增长到 128K 时，计算量（TFLOPs）增长约 122 倍
- **目标**：通过动态剪枝冗余token，减少实际参与计算的token数量，从而降低计算复杂度

### 1.2 理论基础：梯度特征归因（Gradient-based Feature Attribution）
论文基于**梯度特征归因理论**，核心思想是：
- 通过分析输出对输入的偏导数，可以量化每个输入特征（token）的重要性
- 梯度幅度反映了输出对输入变化的敏感性
- 高saliency score的token对模型输出的贡献更大

## 二、关键理论观察

### 2.1 Token稀疏性假设
基于对LLM模型的saliency map分析，论文发现两个重要模式：

1. **稀疏性递增**：重要token是稀疏的，且稀疏率随层数增加而增加
   - 定义：saliency score > 10%最大值的token为重要token
   - 观察：随着网络加深，冗余token比例增加

2. **稀疏性传递性**：如果前层token被判定为冗余，在深层仍然冗余
   - 数学表达：若 $\hat{\pi}_i^{(l)} < \theta$，则 $\hat{\pi}_i^{(l+k)} < \theta'$ 的概率很高
   - 这为分层剪枝提供了理论依据

## 三、数学框架

### 3.1 Saliency Score计算

对于第 $s$ 个pruning stage，saliency score定义为：

$$\hat{\pi} = \frac{\partial T(x)}{\partial x} \cdot x$$

其中：
- $T(x)$ 是整个网络的输出
- $x \in \mathbb{R}^N$ 是第 $s$ 个pruning stage所在层的输入向量
- $N$ 是当前token数量
- $\hat{\pi} \in \mathbb{R}^N$ 表示每个token的重要性分数

**数学意义**：
- 这是**梯度×输入**的特征归因方法（类似Integrated Gradients）
- $\frac{\partial T(x)}{\partial x}$ 表示输出对输入的敏感度
- 与输入 $x$ 的逐元素乘积得到每个token的贡献度

### 3.2 Token Pruning Module架构

Pruning module是一个轻量级的两层MLP：

$$\pi = \text{MLP}(\text{GELU}(\text{MLP}(\mathbf{x})))$$

其中 $\pi \in \mathbb{R}^{N \times 2}$ 是pruning module的输出（每个token对应2个logits）。

**决策掩码生成**：
$$M = \text{Gumbel-Softmax}(\pi)$$

其中 $M \in \{0,1\}^N$ 是one-hot向量，表示哪些token需要保留。

**为什么使用Gumbel-Softmax**：
- 提供可微分的离散采样，使得端到端训练成为可能
- 在训练时保持梯度流，在推理时退化为hard decision

### 3.3 损失函数设计

总损失函数由三部分组成：

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{mse}} + \mathcal{L}_{\text{r}}$$

#### 3.3.1 分类损失（Language Modeling Loss）
$$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(y, \hat{y})$$

保持模型的预训练能力，确保剪枝后模型仍能正确预测。

#### 3.3.2 MSE损失（值匹配）
$$\mathcal{L}_{\text{mse}} = \text{MSE}(\pi, \hat{\pi})$$

使pruning module预测的重要性分数 $\pi$ 与真实的saliency score $\hat{\pi}$ 在数值上接近。

#### 3.3.3 Ranking损失（排序匹配）⭐核心创新

总ranking损失是所有stage的ranking损失之和：

$$\mathcal{L}_r = \sum_{s=1}^S \mathcal{L}_r^{(s)}$$

其中 $S$ 是pruning stage的总数。

每个stage的ranking损失定义为：

$$\mathcal{L}_{\mathrm{r}}^{(s)}(\pi, \hat{\pi}) = \sum_{i=1}^{N-1} \sum_{j=i+1}^N \log \left(1+e^{-\left(\left(\pi_i-\pi_j\right) \cdot \operatorname{sign}\left(\hat{\pi}_i-\hat{\pi}_j\right)\right)}\right)$$

**数学解释**：
- 这是**成对排序损失**（pairwise ranking loss）
- 对于所有token对 $(i,j)$：
  - 如果 $\hat{\pi}_i > \hat{\pi}_j$（真实saliency中 $i$ 更重要），则希望 $\pi_i > \pi_j$
  - $\operatorname{sign}(\hat{\pi}_i-\hat{\pi}_j)$ 确保损失函数只在排序方向正确时最小化
- 使用logistic loss形式，使得排序差异越大，损失越大

**为什么需要Ranking Loss**：
- MSE只保证数值接近，但剪枝时我们更关心**相对排序**
- 我们保留高重要性token，剪枝低重要性token
- 即使 $\pi$ 和 $\hat{\pi}$ 的数值有偏差，只要排序正确，剪枝结果仍然正确

## 四、训练策略

### 4.1 两阶段训练方法

#### Stage 1: Token Marking（标记阶段）
- 使用原始模型和所有token进行前向传播
- 计算saliency scores $\hat{\pi}$（需要梯度计算）
- 这些saliency scores作为监督信号

#### Stage 2: Token Pruning（剪枝阶段）
- Pruning module生效
- 使用MSE loss和Ranking loss监督pruning module学习
- 使用Cross-entropy loss保持模型性能
- 端到端优化

### 4.2 分层剪枝策略

**几何序列剪枝率**：
- 设置 $S=10$ 个pruning stages
- 每个stage的保留率 $r_s = r^s$，其中 $r=0.9$
- 即：$[r, r^2, r^3, \ldots, r^{10}] = [0.9, 0.81, 0.729, \ldots]$

**累积剪枝率**：
- 经过10个stages后，累积保留率约为 $r^{10} \approx 0.35$
- 即剪枝约65%的tokens

**为什么从第4层开始**：
- 实验发现：从第4层开始剪枝，保持率90%时效果最好
- 早期层（1-3层）包含更多基础特征，不宜过度剪枝
- 深层（4层以后）特征更抽象，冗余信息更多，可容忍更高剪枝率

## 五、理论优势

### 5.1 计算效率
- Pruning module的计算量 < 1% 的LLM计算量
- 通过减少token数量，直接降低后续层的计算复杂度
- 对于长度为 $N$ 的序列，剪枝到 $rN$ 个token后：
  - Attention复杂度：$O((rN)^2) = O(r^2 N^2)$
  - FLOPs减少：约 $1-r^2$（当 $r=0.35$ 时，约减少88%）

### 5.2 泛化能力
- 冻结所有Transformer blocks，只训练pruning module
- 保持预训练模型的泛化能力
- 只需低资源对齐训练即可获得良好性能

### 5.3 正交性
- Token pruning主要减少prefill阶段的FLOPs
- 与KV cache compression（减少decode阶段）正交
- 可以组合使用，进一步加速

## 六、关键数学洞察

1. **稀疏性传递性**：为分层剪枝提供了理论基础
2. **排序比数值更重要**：Ranking loss确保相对重要性正确
3. **几何序列剪枝**：渐进式剪枝比一次性剪枝更稳定
4. **梯度归因指导**：利用模型自身的梯度信息，而非启发式规则

## 七、实现要点

### 7.1 推理时策略
- 始终保留前4个初始tokens
- 保留局部10%的tokens（滑动窗口）
- 根据pruning module的预测分数选择top-k tokens

### 7.2 训练稳定性
- 使用attention masking模拟token剪枝（类似DynamicViT）
- 避免hard pruning导致的梯度不连续问题

---

**总结**：SDTP的核心是通过梯度特征归因理论，设计可学习的pruning module来预测token重要性，并通过MSE+Ranking双重损失确保预测的准确性。分层渐进式剪枝策略充分利用了token稀疏性的传递性，实现了高效且低损失的token剪枝。

