第一步完成总结
1. 目录结构
已创建目录：
SDTP/  data/    LongBench_data/      ├── narrativeqa.json      (200 items)      ├── qasper.json           (200 items)      ├── gov_report.json       (200 items)      ├── multi_news.json       (200 items)      ├── multifieldqa_en.json  (150 items)      ├── hotpotqa.json         (200 items)      ├── musique.json          (200 items)      ├── triviaqa.json         (200 items)      └── README.md
2. 数据集状态
已成功下载并转换 8 个数据集：
所有文件格式正确
符合 SDTP 格式要求
已验证与 SDTP 加载器兼容
注意：legal_contract_qa 不在 LongBench v1 中。可用的数据集包括：narrativeqa, qasper, multifieldqa_en, multifieldqa_zh, hotpotqa, 2wikimqa, musique, dureader, gov_report, qmsum, multi_news, vcsum, trec, triviaqa, samsum, lsht, passage_count, passage_retrieval_en, passage_retrieval_zh, lcc, repobench-p
3. JSON 格式验证
所有文件格式：
类型：List[Dict]
必需字段：input (string), answers (list of strings)
编码：UTF-8，无 BOM
与 src/evaluation/longbench/dataset.py 兼容
4. 测试命令
运行以下命令测试设置阶段（不执行推理）：
测试 LongBench 评估框架：
# 使用 narrativeqa 数据集bash scripts/run_longbench_setup.sh data/LongBench_data/narrativeqa.json# 使用其他数据集bash scripts/run_longbench_setup.sh data/LongBench_data/hotpotqa.jsonbash scripts/run_longbench_setup.sh data/LongBench_data/qasper.json
测试 LM-Eval Harness 框架：
# Baseline 模型bash scripts/run_lmeval_setup.sh data/LongBench_data/narrativeqa.json baseline# SDTP 模型bash scripts/run_lmeval_setup.sh data/LongBench_data/narrativeqa.json sdtp# 使用其他数据集bash scripts/run_lmeval_setup.sh data/LongBench_data/hotpotqa.json baseline
验证数据集格式：
# 验证所有下载的文件python3 scripts/validate_longbench_data.py
5. 相关脚本
已创建的脚本：
scripts/download_longbench_v1.py - 下载和转换 LongBench v1 数据集
scripts/validate_longbench_data.py - 验证 JSON 格式
使用方法：
# 重新下载所有数据集（覆盖现有文件）python3 scripts/download_longbench_v1.py --overwrite# 下载新数据集（跳过已存在的文件）python3 scripts/download_longbench_v1.py
6. 注意事项
路径更新：脚本默认使用 data/LongBench/narrativeqa.json，但数据集在 data/LongBench_data/。运行测试时请使用完整路径，例如：
   bash scripts/run_longbench_setup.sh data/LongBench_data/narrativeqa.json
数据集数量：每个数据集包含 150-200 个测试样本（这是 LongBench v1 测试集的默认大小）
格式兼容性：已测试并与 SDTP 的 LongBenchDataset 类兼容
7. 最终目录树
SDTP/  data/    LongBench_data/      ├── narrativeqa.json      ├── qasper.json      ├── gov_report.json      ├── multi_news.json      ├── multifieldqa_en.json      ├── hotpotqa.json      ├── musique.json      ├── triviaqa.json      └── README.md
所有数据集已准备就绪，可用于 SDTP 评估管道



