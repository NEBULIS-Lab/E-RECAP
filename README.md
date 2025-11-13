下面是你自己之前给我的内容：下面是 一份精炼、纯文本、不带格式 的总结，用来方便你之后提醒我进度，也方便你在服务器上先检查环境、运行已经完成的内容。

——— 复制以下内容即可 ———

SDTP 复现进度总结（纯文本版）

一、已经完成并且已经给出 Prompt 的部分（这些另一个 AI 可以完全生成代码）
	1.	stage1_saliency.py
	•	含 Dolly 加载、随机采样、hook 注册、grad×activation、saliency.pt 保存等完整逻辑
	2.	stage2_pruning.py
	•	含剪枝模块 MLP、Gumbel、loss（lm/mse/rank）、top-k、冻结 Qwen 主模型等
	3.	sdtp_model.py
	•	整合 Qwen2-7B + 剪枝模块
	•	hidden_states/attention_mask/kv 剪枝
	•	FlashAttention fallback
	•	prune 或不 prune 的 forward 逻辑
	4.	inference_sdtp.py
	•	baseline vs SDTP 推理
	•	Prefill 和 End-to-End profiling
	•	generate 模式
	•	输出 latency/FLOPs
	•	CLI 参数
	5.	run_stage1.sh / run_stage2.sh / run_inference.sh
	•	单卡 torchrun 调用
	•	日志路径
	•	profile 参数
	6.	安装与环境检查脚本
	•	install.sh（完整依赖安装）
	•	check_full_env.sh（依赖、GPU、CUDA、FlashAttention、NCCL、Python pkg 全检查）

这些文件的 Prompt 已经完全准备好，可以随时交给另一个 AI 生成源码。

二、还没有生成 Prompt 的部分（后续要做，但非 Stage1/2 核心）
	1.	longbench_eval.py（长上下文评测）
	2.	lmeval_runner.py（lm-eval-harness 5-shot）
	3.	ablation.py（消融测试脚本）
	4.	plot_latency.py 或 notebooks/analysis.ipynb（绘图与可视化）
	5.	README.md 自动生成 Prompt
	6.	reports/final_summary.md（最终复现报告模板）
	7.	package_results.sh（结果归档脚本）

这些尚未交付 Prompt，可以等确认环境和前期训练无误后再生成。

三、当前工作建议
你现在要做的是：
	1.	用 check_full_env.sh 检查服务器环境
	2.	配置 Qwen2-7B-Instruct 软链接
	3.	下载 Dolly 15k（只需要 data/raw/dolly15k）
	4.	运行 stage1_saliency.py（用 run_stage1.sh）
	5.	检查 saliency.pt 是否生成、维度是否正确
	6.	确认 stage2_pruning.py 能加载 saliency.pt 并执行 dry-run

确认这些工作正常后，再继续让另一个 AI 生成 LongBench、lm-eval、消融、可视化等剩下的脚本。
