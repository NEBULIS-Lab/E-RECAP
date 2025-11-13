# SDTP: Selective Dynamic Token Pruning for Large Language Models

本项目实现了论文 "Saliency-driven Dynamic Token Pruning for Large Language Models" 的复现，基于 Qwen2-7B 模型。

## 项目结构

```
SDTP/
├── checkpoints/          # 模型权重和检查点
│   ├── pruning_module.pt    # Stage 2 训练得到的 Token Pruner 模块
│   ├── saliency.pt          # Stage 1 生成的 token saliency baseline
│   └── qwen2-7b-instruct/   # Qwen2-7B 模型权重（需单独下载）
│
├── data/                 # 数据集
│   └── raw/                 # 原始数据文件（如 Dolly-15k）
│
├── results/              # 实验结果和报告
│   ├── fig/                 # 可视化图表
│   └── part1_sum.md         # 阶段1总结报告（已完成）
│
├── scripts/              # 执行脚本
│   ├── run_stage1.sh        # Stage 1: Saliency 计算
│   ├── run_stage2.sh        # Stage 2: 剪枝模块训练
│   ├── run_inference.sh     # 单 GPU 推理测试
│   ├── run_inference_multigpu.sh  # 多 GPU 推理测试
│   ├── check_full_env.sh    # 环境检查脚本
│   └── install.sh           # 依赖安装脚本
│
└── src/                  # 源代码
    ├── stage1_saliency.py        # Stage 1: 梯度 × 隐状态计算 saliency
    ├── stage2_pruning.py         # Stage 2: 训练可学习的 Token Pruning 模块
    ├── sdtp_model.py            # 核心模型封装，提供剪枝逻辑接口
    ├── inference_sdtp.py        # 单 GPU 推理实现
    ├── inference_sdtp_multigpu.py  # 多 GPU 推理实现
    └── multigpu_test.py          # 多卡显存压力测试
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1+
- NVIDIA GPU (推荐 RTX 5880 或更高，≥24GB VRAM)
- 至少 50GB 可用磁盘空间用于模型存储

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行流程

1. **Stage 1: Saliency 计算**
   ```bash
   bash scripts/run_stage1.sh
   ```

2. **Stage 2: 剪枝模块训练**
   ```bash
   bash scripts/run_stage2.sh
   ```

3. **推理测试**
   ```bash
   # 单 GPU
   bash scripts/run_inference.sh
   
   # 多 GPU
   bash scripts/run_inference_multigpu.sh
   ```

## 阶段1完成总结

阶段1已完成 SDTP 方法的复现，包括：
- ✅ Saliency baseline 计算
- ✅ Token Pruner 模块训练
- ✅ 单 GPU 推理加速（2.6-3× speedup）
- ✅ 多 GPU 推理加速（8-10× speedup）

详细实验结果和分析请参考：[阶段1总结报告](results/part1_sum.md)

## 主要特性

- **动态 Token 剪枝**：在预填充阶段动态移除冗余 token，减少计算量
- **分层剪枝策略**：在 Transformer 的多个层中逐步剪枝，保持模型性能
- **多 GPU 支持**：支持自动分布式推理，显著提升长序列处理速度
- **可学习的重要性预测器**：使用轻量级 MLP 预测 token 重要性，替代传统启发式方法

## 实验结果

- **单 GPU**：Prefill 阶段加速 2.6-3.0×
- **多 GPU (8× RTX 5880)**：端到端加速最高 10×
- **显存节省**：最高可节省 34% GPU 内存
- **性能保持**：在剪枝 65% token 的情况下，保持与原模型相当的性能

## 引用

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@article{sdtp2024,
  title={Saliency-driven Dynamic Token Pruning for Large Language Models},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

本项目仅供研究和教育用途。

