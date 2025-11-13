# SDTP 复现路线（基于 Qwen2-7B）

## 0. 总览与里程碑
- **目标**：在 Qwen2-7B 上复现 Saliency-driven Dynamic Token Pruning (SDTP)，并验证其在长上下文与多任务上的加速与性能保持效果。
- **关键阶段**：
  1. 环境与项目结构搭建
  2. 数据与模型准备
  3. Saliency 标注阶段实现
  4. 动态剪枝模块训练与集成
  5. 推理加速与 Profiling
  6. 评估与消融（含 LongBench、lm-eval、剪枝策略）
  7. 文档与复现报告整理

---

## 1. 环境准备
- **SSH 与目录**：
  1. 登录：`ssh user2@bld-Rack-Server`
  2. 进入项目：`cd /data/private/user2/workspace/SDTP`
  3. 视需要使用 `tmux new -s sdtp` 维持长时间会话（可选）。
- **硬件检测**：逐条执行以下指令确认 GPU 与 CUDA 工具链状态（输出需保存便于定位问题）：
  - `nvidia-smi`：确认 8× NVIDIA 5880 Ada、驱动版本、显存占用。
  - `nvcc --version`：检查 CUDA Toolkit 版本。
  - `which mpirun && mpirun --version`：确认 MPI（若使用 DeepSpeed ZeRO Offload）。
  - `nvidia-smi topo -m`：了解 GPU 拓扑（调优通信时参考）。
- **系统依赖检查**：
  - `for pkg in gcc g++ make git wget; do which $pkg || echo "$pkg missing"; done`
  - 可选安装 `tmux`；无需额外的 MPI / NCCL / 多机依赖。
- **Python 环境**：
  - 直接使用系统自带 `python3.10 -m pip`；如需隔离，可选 `python3.10 -m venv ~/.venv/sdtp_qwen && source ~/.venv/sdtp_qwen/bin/activate`。
  - 默认使用官方源；如需加速或备份，可再配置镜像：  
    `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
  - 安装依赖（根据实际 CUDA 选择 wheel）：  
    `pip install --upgrade pip`  
    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`  
    `pip install transformers datasets accelerate peft bitsandbytes ninja`  
    `pip install deepspeed==0.13.5`  
    `pip install flash-attn --no-build-isolation`（需 CUDA Toolkit ≥ 11.7；升级后为 12.x 可直接安装）  
    `pip install xformers`  
    `pip install numpy scipy scikit-learn pandas matplotlib tqdm pyyaml tensorboard`
- **数据/模型下载模式**：
  - 当前可访问全球网络，可直接使用官方 `pip`、`huggingface_hub`, `wget`。
  - 如需离线备用，可设置：  
    `export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`  
    `export HF_HOME=/data/private/user2/workspace/SDTP/hf_cache`  
    并提前缓存模型与数据。
- **环境校验脚本**：在 `scripts/check_env.sh` 中写入：
  ```
  #!/bin/bash
  nvidia-smi
  python - <<'PY'
  import torch
  print("Torch CUDA available:", torch.cuda.is_available())
  print("GPU count:", torch.cuda.device_count())
  try:
      import flash_attn
      print("FlashAttention imported:", flash_attn.__version__)
  except ImportError as e:
      print("FlashAttention import failed:", e)
  PY
  ```
  赋权并执行：`chmod +x scripts/check_env.sh && ./scripts/check_env.sh`
- **运行方式建议**：单 GPU 足以完成 Stage1/Stage2，后续命令统一使用 `torchrun --nproc_per_node=1`（或直接 `python`），无需配置多机多卡通信。

---

## 2. 项目结构规划（精简版）
为加快迭代，前期只保留最小必要文件，待复现完成后再考虑模块化拆分。
```
SDTP/
├── data/                   # 原始与中间数据
├── logs/                   # 训练/推理日志
├── checkpoints/            # 剪枝模块与中间权重
├── src/
│   ├── sdtp_model.py       # Qwen2-7B + SDTP 模块封装 & 推理逻辑
│   ├── stage1_saliency.py  # Saliency 计算脚本
│   ├── stage2_pruning.py   # 剪枝模块训练脚本
│   └── inference_sdtp.py   # 评估/加速测试入口
└── scripts/
    ├── check_env.sh        # 环境检查
    ├── run_stage1.sh       # 调用 stage1_saliency.py
    ├── run_stage2.sh       # 调用 stage2_pruning.py
    └── run_inference.sh    # 调用 inference_sdtp.py
```
- `README.md`：记录关键命令与结果。
- `requirements.txt`：可在环境稳定后补充（当前直接使用 `pip list` 记录版本）。
- 初期无需 `configs/`、`evaluation/`、`utils/` 等子目录，所有逻辑集中于四个核心脚本中，减少跳转与调试复杂度。
- 如后续需要开源或扩展，再将 `sdtp_model.py` 拆分为模型、剪枝模块、工具函数等子模块。

---

## 3. 数据与模型准备
- **训练集**：对齐论文，使用 `databricks/databricks-dolly-15k`。当前网络可直接执行：  
  `python -c "from datasets import load_dataset; load_dataset('databricks/databricks-dolly-15k').save_to_disk('data/raw/dolly15k')"`  
  Stage1 仅需从中抽样约 1k 条指令；Stage2 同样使用该子集即可。
- **评估集**：
  - `LongBench` 英文子集（single/multi-doc QA、summarization、few-shot、code）
  - `lm-eval-harness` 对应的 5-shot 任务：COPA, PIQA, Winogrande, MathQA, BoolQ, CB, WiC, WSC
- **数据处理**：实现通用数据管线：
  - Tokenizer 选用 Qwen2-7B 对应的 tokenizer；
  - Stage1/Stage2 训练数据构建（prompt + completion）
  - 长上下文测试集生成固定长度 prompt（4K~128K, 对齐 Table 2）。
- **最小化实现**：
  - `scripts/download_data.sh`（可选）：单纯调用 datasets 保存到 `data/raw/`，或直接手动运行上述命令。
  - `stage1_saliency.py` 内部直接使用 `datasets.load_from_disk('data/raw/dolly15k')`，随后随机采样 1k 条并截断到 `seq_len=512~1024`。
- **完成标准**：
  - `data/raw/dolly15k` 可用；如需 LongBench / lm-eval 时再下载。
  - 采样脚本输出日志，明确实际使用的数据量与平均长度。

### 3.2 模型
- 权重当前位于 `/home/user2/.cache/modelscope/hub/models/qwen/Qwen2-7B-Instruct/`，包含 `config.json`, `generation_config.json`, `model-0000x-of-00004.safetensors`, `tokenizer.json` 等文件。
- 建议在项目目录创建软链接或副本：  
  `ln -s /home/user2/.cache/modelscope/hub/models/qwen/Qwen2-7B-Instruct checkpoints/qwen2-7b-instruct`  
  或 `cp -r /home/user2/.cache/modelscope/hub/models/qwen/Qwen2-7B-Instruct checkpoints/qwen2-7b-instruct`
- 加载时：  
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      \"checkpoints/qwen2-7b-instruct\",
      local_files_only=True,
      trust_remote_code=True,
      device_map=\"auto\"
  )
  tokenizer = AutoTokenizer.from_pretrained(
      \"checkpoints/qwen2-7b-instruct\",
      local_files_only=True,
      trust_remote_code=True,
      use_fast=False
  )
  ```
- 若需 LoRA/Adapter：在 `configs/` 添加 `lora.yaml`，默认禁用；如显存不足再启用。

---

## 4. Stage 1：Saliency 标注实现
- **目标**：在不剪枝的前提下，对选定层的隐藏状态计算 saliency 分数，产出一个用于 Stage2 的监督文件。
- **实现重点**：
  - 在 `src/stage1_saliency.py` 中直接完成数据加载、前向、反向与保存，无需额外模块。
  - 从 Dolly 15K 中随机采样 ~1000 条指令，截断到 `seq_len=512`（如显存余量可调至 1024）。
  - 选择 10 个剪枝层索引（与论文一致），在这些层上做 hook。
  - 单次前向即可计算 saliency：  
    `saliency = (grad * hidden_states).sum(dim=-1)`  
    使用 `torch.autograd.grad` 或 `loss.backward()` 后读取缓存的梯度。
  - 将所有样本的 saliency 拼接成一个字典并保存为 `checkpoints/saliency.pt`（包含：`layer_index`, `token_scores`, `attention_mask` 等）。
- **脚本结构（示意）**：
  - `prepare_data()`：返回采样后的 prompt/label。
  - `register_hooks(model, target_layers)`：捕获前向隐藏状态与梯度。
  - `compute_saliency()`：循环样本，执行前向和反向；可使用 `torch.cuda.amp.autocast()`。
  - `save_saliency(output_path, saliency_dict)`。
- **运行建议**：
  - 使用单 GPU，`batch_size=1~2`，无须梯度累积。
  - 仅执行一次 pass，总耗时约数小时以内。
- **完成标准**：
  - 生成 `checkpoints/saliency.pt`（体积在数百 MB 以内）。
  - 日志记录：采样数量、层索引、平均 saliency 稀疏度。
  - README 更新运行命令（如 `bash scripts/run_stage1.sh`）。

### 4.1 Qwen2 Hook 细节（极重要）
- Qwen2 的 block 结构为 `model.model.layers[i]`，其中 `forward` 返回 `(hidden_states, present)`。
- 推荐 hook 的位置：
  ```python
  def create_hooks(layer_idx):
      def forward_hook(module, input, output):
          cache["hidden_states"][layer_idx] = output[0].detach()
      def backward_hook(module, grad_input, grad_output):
          cache["grads"][layer_idx] = grad_output[0].detach()
      return forward_hook, backward_hook
  ```
  将 hook 注册在 `model.model.layers[i]` 上，确保 `output[0]`/`grad_output[0]` 与隐藏状态对应。
- 如果仅 hook MLP/gate 层，维度会不匹配；务必在 block 返回处拦截。
- saliency 计算示例：
  ```python
  scores = (cache["hidden_states"][i] * cache["grads"][i]).sum(dim=-1)
  ```
- 采样结束后记得 `model.zero_grad(set_to_none=True)`，避免梯度残留。

---

## 5. Stage 2：动态剪枝模块训练
- **核心思想**：冻结 Qwen2-7B 主模型，仅训练一个轻量级的 token 评分模块，让其逼近 Stage1 提供的 saliency 排序。
- **实现位置**：在 `src/stage2_pruning.py` 中完成全部逻辑；剪枝模块类可写在同文件或 `src/sdtp_model.py` 中的辅助类。
- **模块设计**：
  - 两层 MLP + GELU，输出每个 token 的 logits（shape `[B, T]`）。
  - 使用 Gumbel-Softmax/Concrete 分布生成近似二值 mask；支持设定保留率（默认几何序列 `r=0.9`，共 10 层）。
  - 训练损失 = `L_cls`（语言建模） + `L_mse`（与 saliency 对齐） + `L_rank`（pairwise ranking）。
- **训练流程**：
  1. 加载 `checkpoints/saliency.pt`，按 batch 提供 `hidden_states` 对应的目标分数。
  2. 在 `sdtp_model.py` 中集成剪枝模块：前向时先生成 mask，再对当前层的 hidden states / attention_mask / kv 缓存进行裁剪。
  3. 对 Dolly 子集继续进行语言建模训练；主模型参数 `requires_grad=False`。
  4. 训练可在单卡上完成，`batch_size=1~2`，`num_epochs=2` 与论文一致。
- **保留率处理**：
  - 建议在代码中定义 `keep_ratios = [r ** i for i in range(len(target_layers))]`。
  - 可在训练日志打印实际保留 token 数，便于对齐论文（65% 剪枝）。
- **输出**：
  - 保存 `checkpoints/pruning_module.pt`（内含剪枝模块参数、层配置、keep_ratio）。
  - 写入一个小型 `json`（如 `pruning_config.json`）说明剪枝层索引、保留率，供推理阶段加载。
- **监控指标**：
  - 训练损失曲线（尤其是 `L_rank` 收敛情况）。
  - Dolly 验证集 perplexity（与未剪枝模型对比，仅需少量 batch）。
  - 可选：统计剪枝前后 token 保留比例的分布。

### 5.1 Pruning Mask 传播要点
- **裁剪对象**：`hidden_states`, `attention_mask`, `position_ids`, `past_key_values`（prefill）。
- **实现顺序**：
  1. 根据 `keep_ratio` 选出保留的 token 索引（按 scores top-k）。
  2. 对当前层的 `hidden_states` 做 `index_select`。
  3. 更新 `attention_mask`：需同步裁剪对应 token，并保持形状 `[B, 1, 1, T]`。
  4. Prefill 阶段的 `past_key_values`：在裁剪前 key/value 尚未缓存，可直接构建新的 kv；若复用缓存，需同步裁剪其第二维。
  5. 维护一个 `token_index` 列表，供后续层继续使用，防止 reshape 混乱。
- **Gumbel-Softmax 注意**：
  - 训练阶段可使用 `torch.nn.functional.gumbel_softmax(logits, tau, hard=True)`。
  - 推理阶段直接使用 `torch.topk` 得到硬索引，避免随机性。
- **调试技巧**：
  - 每层打印 `retained_tokens`，确认与 keep_ratio 匹配。
  - 若出现 tensor shape 报错，优先检查 attention_mask 是否同步裁剪。

---

## 6. 推理集成与加速评估
- **推理管线**：
  - 在 `src/sdtp_model.py` 中实现 `forward(prune=True)`：根据保留率在每个目标层裁剪 hidden_states / attention_mask / kv。
  - `src/inference_sdtp.py` 负责加载基础模型与剪枝模块，提供三种模式：  
    `--mode generate`（常规生成）  
    `--mode profile`（测 Prefill / End2End 时间）  
    `--mode compare`（对比剪枝前后输出）
  - Prefill 时可继续使用 FlashAttention；若遇到 shape 不匹配，回退到 PyTorch 原生注意力实现（计划内的权衡：正确性优先，速度后续优化）。
- **运行方式**：
  - 单卡即可，命令示例：`bash scripts/run_inference.sh --lengths 4096 8192 16384`.
  - `run_inference.sh` 内部调用 `python src/inference_sdtp.py --keep_profile default --max_new_tokens 128`.
- **完成标准**：
  - 输出 `logs/profile/latency_{timestamp}.json`（记录 prompt 长度、保留率、Prefill/End2End latency、FLOPs 估算）。
  - 保存示例生成文本 `logs/samples/sample_*.txt`，用于人工校验质量。
  - README 中总结剪枝前后速度对比表。

### 6.1 Attention 实现注意
- Qwen2 默认使用 FlashAttention；剪枝后 token 数随层变化，FlashAttention 可能因 shape 与 stride 不匹配报错。
- 安全方案：
  1. 在 `sdtp_model.py` 中根据 `prune` 状态切换实现：  
     ```python
     if use_flash and tokens_are_contiguous:
         attn_out = flash_attn_func(q, k, v, ...)
     else:
         attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
         attn_scores = attn_scores + attention_mask
         attn_out = torch.matmul(attn_probs, v)
     ```
  2. 保证裁剪后的 `k/v` shape 与 `q` 对齐，并在 fallback 时禁用 FlashAttention（速度稍慢但结果正确）。
- Prefill/解码阶段区分处理：prefill 剪枝后 `past_key_values` 需要更新；decode 阶段可保持原逻辑（本方案主要影响 Prefill）。
- 若需要重新启用 FlashAttention，可在未来阶段优化索引映射，但首次复现以功能正确为先。
- **Profiling 指标**：
  - Prefill/Decode 阶段耗时、总耗时、FLOPs 估算、显存峰值；
  - 不同序列长度（4K, 8K, 16K, 32K, 64K, 128K）的速度曲线；
  - 剪枝模块自身开销占比。
- **工具**：`torch.utils.benchmark`, `nvprof`/`nsys`, 自行统计 FLOPs（基于 attention/MLP token 数减少）。
- **指标定义与计算**：
  - `Prefill Latency (s)`: 从输入 prompt 送入到生成首个 token 前的时间，使用 `torch.cuda.Event` 或 `time.perf_counter` + `torch.cuda.synchronize()` 精确测量，反映剪枝对高算力阶段的提升。
  - `End-to-End Latency (s)`: 完整生成固定输出长度（默认 128 token）的总耗时，体现用户体验层面的加速。
  - `FLOPs Reduction (%)`: 基于每层保留 token 数估算注意力和 FFN FLOPs，公式示例：  
    `FLOPs_pruned = Σ_l [4 * (T_l^2 * d) + 8 * (T_l * d^2)]`，其中 `T_l` 为第 `l` 层保留 token 数；与原始 FLOPs 比较得出降低比例。
  - `Speedup Ratio`: `Latency_baseline / Latency_pruned`，分别对 Prefill 与 End-to-End 计算。
  - `Compression Ratio (1/τ)`: `原始 token 数 / 剪枝后 token 数`。在多层剪枝时，统计各层平均保留 token 数，取 `prompt_length / mean(T_l)`。
  - `GPU Memory (GB)`: 使用 `torch.cuda.max_memory_allocated()` 或 `nvidia-smi --query-gpu=memory.used` 记录峰值，用于评估显存节省。

---

## 7. 评估实验设计
- **指标说明**：
  - `任务得分`: LongBench/lm-eval 官方指标（如 EM、Rouge-L、Accuracy），反映语义能力是否保持。
  - `Compression Ratio`: token 压缩倍数，衡量剪枝力度；需与任务得分一起观察权衡。
  - `Speedup Ratio`: Prefill/End-to-End 的时间加速比，直接体现推理效率改进。
  - `FLOPs Reduction` & `Memory Saving`: 分别说明算力与显存开销的下降幅度；帮助判断在不同硬件配置下的收益。
  - 记录上述指标，可对照原论文表 1/2，验证复现质量。
- **待实现/更新文件**：
  - `src/evaluation/longbench_eval.py`：加载离线 LongBench 数据，输出表格 `results/longbench_{profile}.csv`。
  - `src/evaluation/lmeval_runner.py`：与 lm-eval-harness 的离线脚本结合，生成 `results/lmeval_{profile}.json`。
  - `src/evaluation/ablation.py`：批量运行不同剪枝策略，整理对比表。
  - `scripts/run_longbench.sh` / `scripts/run_lmeval.sh` / `scripts/run_ablation.sh`：分别封装关键参数。
- **完成标准**：
  - `results/` 目录下包含 LongBench、lm-eval、消融等 csv/json。
  - README 中同步展示对比表，与论文指标对照。
### 7.1 长上下文 (LongBench)
- 执行长上下文任务评测，记录以下指标：任务得分、压缩比、Prefill 加速、端到端加速。
- 设定两档剪枝强度：`SDTP` (压缩比 ≈1.6) 与 `SDTP-low` (≈1.3)。

### 7.2 多任务泛化 (lm-eval-harness)
- 在 COPA、PIQA、Winogrande、MathQA、BoolQ、CB、WiC、WSC 上进行 5-shot 测试，比较剪枝前后准确率。

### 7.3 消融实验
- 去除排名损失、去除 MSE、单阶段剪枝 vs 多阶段剪枝、不同保留率/起始层。
- 记录每个设置的性能、加速、token 保留曲线。

### 7.4 与 KV Cache 压缩兼容性
- 集成 `H2O` 或其他 KV 压缩策略，测试组合效果，验证无复合误差。

---

## 8. 结果记录与可视化
- 生成对齐论文中的表格（FLOPs/Latency、LongBench 成绩、lm-eval 对比、消融表）。
- 绘制 saliency 稀疏度、首剪枝层影响、多阶段剪枝效果等图像。
- 整理日志、TensorBoard，总结关键指标。
- **待实现/更新文件**：
  - `notebooks/analysis.ipynb` 或 `reports/analysis.md`：可视化曲线、生成论文中对应图表。
  - `reports/final_summary.md`：最终复现总结，包括硬件配置、训练时间、性能差异分析。
- **完成标准**：
  - 图像文件输出到 `reports/figures/`，命名规范 `figure_saliency_layer.png` 等。
  - 最终报告可独立阅读，覆盖方法、实验、结论。

### 8.1 可视化设计建议
- **Saliency 分布**（对应论文 Figure 1/2）：
  - 图表类型：堆叠条形图或热力图，展示不同层的重要 token 占比。
  - 输入数据：Stage1 生成的 saliency，按层统计 `top_k` 稀疏度。
  - 预期输出：`reports/figures/saliency_sparsity_layers.png`
- **剪枝层起点影响**（对应 Figure 4）：
  - 图表类型：折线图，横轴为起始层，纵轴为 Winogrande 等任务准确率。
  - 颜色区分不同保留率 `r=0.9/0.8/0.7`。
  - 输出：`reports/figures/pruning_start_layer.png`
- **多阶段剪枝效果**（对应 Figure 5）：
  - 图表类型：多折线图，横轴为剪枝阶段数，纵轴为各任务准确率，附平均趋势线。
  - 输出：`reports/figures/multi_stage_accuracy.png`
- **Latency & FLOPs**（对应 Table 2 可视化）：
  - 图表类型：双轴图或两个子图；分别展示序列长度 vs. 延迟、FLOPs、显存。
  - 输出：`reports/figures/latency_floops_curve.png`
- **长上下文与 lm-eval 对比**：
  - 图表类型：分组柱状图，比较 Original、SDTP、SDTP-low；可分别生成 LongBench 与 lm-eval 图。
  - 输出：`reports/figures/longbench_comparison.png`, `reports/figures/lmeval_comparison.png`
- **报告嵌入建议**：
  - 在 `reports/final_summary.md` 中为每张图预留说明段落：描述数据来源、观察结论、与原论文一致性。
  - 图像统一使用 `matplotlib`/`seaborn`，设置字体与配色，确保打印友好。

---

## 9. 风险与备选方案
- **显存压力**：如 Stage1 梯度计算过大，可尝试梯度检查点、ZeRO Stage 2/3、降低 batch size。
- **FlashAttention 兼容性**：若 Qwen2-7B 在特定库版本下不稳定，可回退到官方推荐组合。
- **数据不足**：若 Dolly-15k 对 Qwen 调优不足，可增补自定义指令数据或使用合成长上下文数据。
- **性能差异**：如评估指标低于论文，可调整剪枝层位置与保留率，或考虑对主模型少量微调。

---

## 10. 后续文档与发布
- 编写细化的 README 与复现报告：描述环境、配置、命令、实验结果。
- 打包脚本与配置，确保他人可在同类硬件上复现。
- 最终对比与论文差异分析，指出 Qwen2-7B 与原模型的性能差异及原因。
- **待实现/更新文件**：
  - `README.md`：增加 “快速开始”、“环境要求”、“实验结果” 章节。
  - `LICENSE` / `CITATIONS.md`（如需）。
  - `scripts/package_results.sh`：收集关键日志与结果，便于归档。
- **完成标准**：
  - README 提供完整命令流程，可供他人按步执行。
  - 整体仓库结构与文档满足复现要求。

---

## 11. 分阶段任务清单（建议执行顺序）
1. **环境配置**
   - [ ] 执行 `scripts/check_env.sh`，记录 GPU/CUDA 状态
   - [ ] 补充缺失依赖、完成 PyTorch + FlashAttention 安装/验证（失败时记录原因并决定是否 fallback）
   - [ ] 确认单卡训练命令 `torchrun --nproc_per_node=1` 可正常运行
2. **数据与模型准备**
   - [ ] 整理并拷贝 Dolly-15k、LongBench、lm-eval 数据到 `data/raw/`
   - [ ] 在 `stage1_saliency.py` 内实现随机采样逻辑（无需额外预处理目录）
   - [ ] 创建模型软链接 `checkpoints/qwen2-7b-instruct`
   - [ ] 在 `sdtp_model.py` 中测试基础前向（未剪枝）
3. **Stage1 Saliency 计算**
   - [ ] 在 `stage1_saliency.py` 中实现 hook/grad 逻辑
   - [ ] 使用 100~200 条数据 dry-run，确认 `checkpoints/saliency.pt` 结构正确
   - [ ] 扩展到约 1000 条样本并记录统计信息
4. **Stage2 剪枝模块训练**
   - [ ] 在 `sdtp_model.py` 中实现剪枝模块与 mask 传播
   - [ ] 在 `stage2_pruning.py` 中加载 `saliency.pt` 并完成训练流程
   - [ ] 评估 Dolly 子集 perplexity（剪枝 vs 未剪枝）
5. **推理与 Profiling**
   - [ ] 在 `sdtp_model.py` 中实现可切换的注意力逻辑（FlashAttention → PyTorch fallback）
   - [ ] `inference_sdtp.py` 支持 generate/profile 两种模式
   - [ ] 使用 `scripts/run_inference.sh` 收集 4K~32K 序列的 latency/FLOPs
6. **评估与消融**
   - [ ] 运行 `scripts/run_longbench.sh` 收集长上下文指标
   - [ ] 运行 `scripts/run_lmeval.sh` 完成 5-shot 泛化测试
   - [ ] 使用 `scripts/run_ablation.sh` 探索不同剪枝策略，整理结果表
7. **可视化与报告**
   - [ ] 在 `notebooks/analysis.ipynb` 中绘制关键曲线/表格
   - [ ] 写作 `reports/final_summary.md`，对比论文指标、总结经验
   - [ ] 更新 `README.md`，确保他人可按文档复现
