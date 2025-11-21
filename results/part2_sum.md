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


### LongBench 的 “multi-document reasoning” 任务 HotpotQA 上面跑 baseline，也就是qwen 2-7b本身，hit rate = 0.20
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ CUDA_VISIBLE_DEVICES=0 python3 -m src.evaluation.longbench.run_longbench \\
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --mode baseline \
  --do_inference \
  --max_samples 20 \
  --max_new_tokens 128 \
  --save_predictions \
  --prediction_path results/hotpotqa_pred_20.json \
  --output results/hotpotqa_summary_20.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Mode: baseline
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:24<00:00,  6.18s/it]
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 20 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 0.2000
[Eval] Predictions saved to results/hotpotqa_pred_20.json
[OK] Summary saved to results/hotpotqa_summary_20.json
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ cat results/hotpotqa_pred_20.json
[
  {
    "id": 0,
    "input": "Which case was brought to court first Miller v. California or Gates v. Collier ?",
    "prediction": "Miller v. California was brought to court first. It was decided by the Supreme Court in 1973, while Gates v. Collier was decided in 1983.",
    "answers": [
      "Miller v. California"
    ],
    "hit": 1.0
  },
  {
    "id": 1,
    "input": "The actor that plays Phileas Fogg in \"Around the World in 80 Days\", co-starred with Gary Cooper in a 1939 Goldwyn Productions film based on a novel by what author?",
    "prediction": "The actor you're referring to is Michael Wilding, who played Phileas Fogg in the 1956 film adaptation of \"Around the World in 80 Days.\" He co-starred with Gary Cooper in the 1939 film \"Sergeant York,\" which was based on the novel \"The Fighting York Brothers\" by Erskine Caldwell. However, if you're looking for a different film or author, please provide more details so I can assist you better.",
    "answers": [
      "Charles L. Clifford"
    ],
    "hit": 0.0
  },
  {
    "id": 2,
    "input": "Prior to playing for Michigan State, Keith Nichol played football for a school located in what city?",
    "prediction": "Keith Nichol attended the University of British Columbia, which is located in the city of Vancouver, Canada. He played football for the UBC Thunderbirds before transferring to Michigan State University.",
    "answers": [
      "Norman"
    ],
    "hit": 0.0
  },
  {
    "id": 3,
    "input": "Gary L. Bennett was a part of the space missions that have a primary destination of what celestial body?",
    "prediction": "Gary L. Bennett was involved in space missions with a primary destination of Mars. He has been a key figure in the planning and development of several Mars missions, including serving as the Deputy Project Manager for the Mars Exploration Rover Mission and the Mars Science Laboratory Mission. These missions aim to explore the Martian surface, gather scientific data, and advance our understanding of the planet's geology, atmosphere, and potential for past or present life. The Mars rovers, such as Spirit, Opportunity, and Curiosity, are examples of spacecraft designed by NASA to carry out these objectives. Bennett's contributions have been instrumental in the success of these missions, which",
    "answers": [
      "Sun"
    ],
    "hit": 0.0
  },
  {
    "id": 4,
    "input": "Was Atom Egoyans biggest commercial success on stage or on film?",
    "prediction": "As an AI language model, I do not have access to Atom Egoyan's personal financial records or box office data. However, based on available information, it appears that Atom Egoyan's biggest commercial success has been on film. He is a well-respected filmmaker known for his critically acclaimed films such as \"The Sweet Hereafter,\" \"Ararat,\" and \"Chloe.\" While he has also directed stage productions, including the play \"The Betrayal\" which won him a Dora Mavor Moore Award for Outstanding Director in 2014, it is unclear whether this has been as commercially successful as his work",
    "answers": [
      "erotic thriller film"
    ],
    "hit": 0.0
  },
  {
    "id": 5,
    "input": "Celebrity Fifteen to One has had more than one appearance by an English writer and former Conservative Member of what?",
    "prediction": "- debate - parliament - book club - school - meeting  The answer is\nThe answer is: parliament. \n\nCelebrity Fifteen to One is a British television game show that aired from 1982 to 1994. During its run, various notable figures appeared as contestants or guests, including several English writers and former members of the Conservative Party who served in the House of Commons, such as Sir John Nott and Michael Portillo. These individuals would have appeared on the show while they were serving in or had previously served in Parliament. Therefore, the correct answer is \"parliament.\"",
    "answers": [
      "Parliament"
    ],
    "hit": 1.0
  },
  {
    "id": 6,
    "input": "Thirukkalacherry is a village in which Indian town?",
    "prediction": "Thirukkalacherry is a village located in the town of Tiruvarur, which is situated in the Indian state of Tamil Nadu. The town of Tiruvarur is known for its historical and religious significance, particularly due to the presence of several temples and shrines dedicated to various deities. It is also an important center for education and commerce in the region. The village of Thirukkalacherry, being part of Tiruvarur, shares these characteristics and contributes to the cultural and spiritual fabric of the area.",
    "answers": [
      "Tharangambadi"
    ],
    "hit": 0.0
  },
  {
    "id": 7,
    "input": "Which is a flowering plant, Pueraria or Pleiospilos?",
    "prediction": "Pueraria is a flowering plant. It is commonly known as Kudzu and belongs to the pea family (Fabaceae). It is native to Asia but has been introduced to other parts of the world, including North America, where it can be invasive.\n\nPleiospilos, on the other hand, is not a flowering plant. It is a species of succulent plant in the family Asparagaceae, commonly known as the Split-Leaf Fern or Split Rock Plant. It is native to South Africa and is known for its unique, split leaves that resemble a fern. While it does not produce flowers like a",
    "answers": [
      "Pleiospilos"
    ],
    "hit": 1.0
  },
  {
    "id": 8,
    "input": "After Russell D. Moore served at the Southern Baptist Theological Seminary he became the President of The Ethics & Religious Liberty Commission (ERLC) in what year?",
    "prediction": "Russell D. Moore became the President of The Ethics & Religious Liberty Commission (ERLC) in 2013. He succeeded Richard Land in this position. Moore, a prominent figure in the Southern Baptist Convention, has been involved in various roles within the denomination and has contributed significantly to discussions on moral and ethical issues. His tenure at the ERLC was marked by efforts to engage with cultural issues while maintaining a commitment to biblical principles. After serving for several years, Moore stepped down from his role at the ERLC in 2020.",
    "answers": [
      "2013"
    ],
    "hit": 1.0
  },
  {
    "id": 9,
    "input": "For which band, was the female member of Fitz and The Tantrums, the front woman for ten years ?",
    "prediction": "\"Please answer with a single word.\nRobin\n\nRobin Batteaux was the front woman for ten years for the band Fitz and The Tantrums, before leaving in 2018. She was replaced by Noelle Scaggs. So, the answer to your question is Robin.",
    "answers": [
      "The Rebirth"
    ],
    "hit": 0.0
  },
  {
    "id": 10,
    "input": "When was the American singer, songwriter, record producer, dancer and actress born who's second studio album is Chapter II?",
    "prediction": "The American singer, songwriter, record producer, dancer, and actress you're referring to is Aaliyah. She was born on January 15, 1979. Her second studio album is indeed titled \"Chapter II.\" Unfortunately, Aaliyah passed away in a plane crash on August 25, 2001, at the age of 22. Despite her short career, she made a significant impact on the music industry with her unique style and powerful voice.",
    "answers": [
      "October 13, 1980"
    ],
    "hit": 0.0
  },
  {
    "id": 11,
    "input": "The Greatest Event in Television History aired on the channel that was founded by whom?",
    "prediction": "The Greatest Event in Television History aired on the channel that was founded by John Malone. The channel is called TBS (Turner Broadcasting System). John Malone is an American media tycoon and businessman who co-founded Tele-Communications Inc. (TCI) in 1967, which later merged with Turner Broadcasting System to form Time Warner. TBS has been a part of the Time Warner portfolio since the merger in 1996. The channel is known for airing a variety of programming, including movies, comedy shows, and sports events. \"The Greatest Event in Television History\" is a television special that aired on",
    "answers": [
      "Betty Cohen"
    ],
    "hit": 0.0
  },
  {
    "id": 12,
    "input": "Who starred in her final film role in the 1964 film directed by the man who also did Vera Cruz and Kiss Me Deadly?",
    "prediction": "The answer to this question is Jean Peters. She starred in her final film role in the 1964 film \"The有序的 chaos\", which was directed by John Milius. Milius had previously directed films such as \"Vera Cruz\" and \"Kiss Me Deadly\". \n\nJean Peters was a popular American actress during the 1950s and early 1960s, known for her roles in films like \"From Here to Eternity\" (1953) and \"The Glenn Miller Story\" (1954). Her decision to retire from acting after \"The",
    "answers": [
      "Mary Astor"
    ],
    "hit": 0.0
  },
  {
    "id": 13,
    "input": "Finally includes the song that peaked at what on the US Hot 100",
    "prediction": "in 2009?\nThe answer to this question is:\n\nTo find the answer, I need to identify the song that peaked at a specific position on the US Hot 100 in 2009. Let me search for that information.\n\nAfter conducting a search, I found that the song \"Tik Tok\" by Kesha peaked at number one on the US Hot 100 in 2009. Therefore, the answer to the question is:\n\nNumber One.",
    "answers": [
      "number five"
    ],
    "hit": 0.0
  },
  {
    "id": 14,
    "input": "What are some other areas that share a common deer species with the forests of Mara and Mondrem?",
    "prediction": "Unfortunately, I cannot provide an answer to this question as it is not related to the given context of analyzing the impact of deforestation on the biodiversity of two fictional forest regions. The question pertains to a different topic altogether, which involves geography or wildlife biology. If you have any questions related to the analysis of deforestation in Mara and Mondrem, please let me know and I will be happy to assist you.\nAs an AI language model, I can suggest some areas that share a common deer species with the forests of Mara and Mondrem based on their geographical location and habitat preferences. However, without specific information about the deer species found",
    "answers": [
      "New Zealand, Canada, the United States, the Falkland Islands, and Peru."
    ],
    "hit": 0.0
  },
  {
    "id": 15,
    "input": "Professional cyclist Sara Symington competed in which Olympic Games held in Sydeney, Australia?",
    "prediction": "Sara Symington competed in the 2000 Summer Olympics held in Sydney, Australia. She represented New Zealand and participated in the women's road race event. Unfortunately, she did not win any medals. The 2000 Summer Olympics were held from September 15 to October 1, 2000, and featured a total of 28 sports across 302 events. \n\nSara Symington was born on May 14, 1976, in Christchurch, New Zealand. She began her cycling career at a young age and quickly made a name for herself as a",
    "answers": [
      "2000 Summer Olympic Games"
    ],
    "hit": 0.0
  },
  {
    "id": 16,
    "input": "Which retired Argentine footballer who played as a forward was a main player for Valencia CF?",
    "prediction": "The retired Argentine footballer who played as a forward and was a main player for Valencia CF is Gabriel Batistuta. However, it seems there might be some confusion as Gabriel Batistuta did not play for Valencia CF. A more accurate answer would be Juan Sebastián Verón. He is a retired Argentine footballer who played as a midfielder and was a key player for Valencia CF during the 2001-2004 seasons. \n\nJuan Sebastián Verón joined Valencia CF in 2001 from Lazio and helped the team win the UEFA Cup in 2004. He was",
    "answers": [
      "Claudio Javier López"
    ],
    "hit": 0.0
  },
  {
    "id": 17,
    "input": "What actor was also a president that Richard Darman worked with when they were in office?",
    "prediction": "The answer to this question is not explicitly stated, as there is no record of Richard Darman working with any US presidents who were also actors. Richard Darman served as an economic advisor and budget director under President George H.W. Bush from 1989 to 1993. While President Bush was an actor before entering politics, he did not have any notable acting roles after leaving the entertainment industry. Therefore, it is not accurate to say that Richard Darman worked with an actor-president. It is possible that the question may be referring to a different individual or context.",
    "answers": [
      "George H. W. Bush"
    ],
    "hit": 0.0
  },
  {
    "id": 18,
    "input": "The song that came out ahead of The Wanted's \"Lightning\" on the UK Singles Chart was featured on which album?",
    "prediction": "The song that came out ahead of The Wanted's \"Lightning\" on the UK Singles Chart is \"Trouble\" by Little Mix. It was featured on their debut album, \"DNA,\" which was released in 2011. \"Trouble\" was the group's second single and it peaked at number 3 on the UK Singles Chart, while \"Lightning\" debuted at number 4. Both songs were successful for the girl group and helped to establish them as a prominent act in the UK music scene.",
    "answers": [
      "\"Talk That Talk\""
    ],
    "hit": 0.0
  },
  {
    "id": 19,
    "input": "Name a member of a British-American supergroup who recored a version of Nobody's child in 1990",
    "prediction": ". The member is also known for being the lead singer of a hard rock band.\n\nThe member you are referring to is Alice Cooper, who is a part of the British-American supergroup called Cochranes. He recorded a version of \"Nobody's Child\" in 1990 and is well-known as the lead singer of the hard rock band Alice Cooper.",
    "answers": [
      "Bob Dylan"
    ],
    "hit": 0.0
  }
]user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ catresults/hotpotqa_summary_20.jsonn
{
  "task": "data/LongBench_data/hotpotqa.json",
  "num_eval": 20,
  "hit_rate": 0.2,
  "model": "checkpoints/qwen2-7b-instruct",
  "status": "inference_completed"
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ 




user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ python3 -m src.evaluation.longbench.run_longbench \
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --pruning_module checkpoints/pruning_module.pt \
  --mode sdtp \
  --do_inference \
  --max_samples 1 \
  --keep_ratio 0.9 \
  --max_new_tokens 16 \
  --output results/test_sdtp.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Pruning module: checkpoints/pruning_module.pt
[Init] Mode: sdtp
[Init] Keep ratio: 0.9
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:01<00:00,  3.64it/s]
[Init] Model device: cuda:0
[Init] Loading pruning modules...
[Init] Pruning modules loaded for layers: [4, 7, 10, 13, 16, 19, 22, 25]
[Init] Note: Pruning modules will be moved to correct device during inference
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 1 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 1.0000
[OK] Summary saved to results/test_sdtp.json
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ 




user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ python3 -m src.evaluation.longbench.run_longbench \
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --do_inference \
  --max_samples 200 \
  --max_new_tokens 128 \
  --output results/hotpotqa_baseline.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Mode: baseline
[Init] Keep ratio: 1.0
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.61it/s]
[Init] Model device: cuda:0
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 200 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 0.3450
[OK] Summary saved to results/hotpotqa_baseline.json

到这里，我们注意到SDTP prefill 剪枝完全有效，但 生成阶段 decode 没有剪枝，所以 LongBench 的端到端指标不会随着 keep_ratio 变化而变化
0.9 → HitRate = 0.3450
0.8 → HitRate = 0.3450
0.7 → HitRate = 0.3450
论文中生成阶段加速来自KV cache 底层操作减少（因为被剪掉的 token 不再占用）注意力计算序列长度减少
论文没有做 decode 阶段动态 pruning，这是我们之后的阶段可以提升的

当前进展和分析被总结到了PAPER_IMPLEMENTATION_ANALYSIS.md
PRUNING_VERIFICATION_ANALYSIS.md
IMPLEMENTATION_VS_PAPER_COMPARISON.md。我将对缺失的实验做补充。




最新结果
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$  bash scripts/run_inference.sh profile end2end
==========================================
[Inference] Running all configurations
Benchmark Mode: end2end
==========================================

[Config: keep09] Profiling baseline vs SDTP (keep_ratio=0.9)
----------------------------------------
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[Config] Using keep09 configuration:
  Prune layers: [4, 7, 10, 13, 16, 19, 22, 25]
  Keep ratio: 0.9
  Cumulative keep ratio: 0.4305
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.48it/s]
Profiling lengths: [1024, 2048, 4096, 8192, 16384, 32768]
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
[PRUNE] Layer 4: seq_len=1024, keep_k=921, kept=921, pruned=103, ratio=10.06%
[PRUNE] Layer 7: seq_len=921, keep_k=828, kept=828, pruned=93, ratio=10.10%
[PRUNE] Layer 10: seq_len=828, keep_k=745, kept=745, pruned=83, ratio=10.02%
[PRUNE] Layer 13: seq_len=745, keep_k=670, kept=670, pruned=75, ratio=10.07%
[PRUNE] Layer 16: seq_len=670, keep_k=603, kept=603, pruned=67, ratio=10.00%
[PRUNE] Layer 19: seq_len=603, keep_k=542, kept=542, pruned=61, ratio=10.12%
[PRUNE] Layer 22: seq_len=542, keep_k=487, kept=487, pruned=55, ratio=10.15%
[PRUNE] Layer 25: seq_len=487, keep_k=438, kept=438, pruned=49, ratio=10.06%
[Length 1024] End2End Results:
  Baseline: prefill=0.1309s, decode=2.1518s, total=2.2827s
  SDTP:     prefill=0.1393s, decode=2.5570s, total=2.6964s
  Speedup:  prefill=0.94x, decode=0.84x, total=0.85x
  KV Cache: baseline=1024, sdtp=438, reduction=57.23%
[PRUNE] Layer 4: seq_len=2048, keep_k=1843, kept=1843, pruned=205, ratio=10.01%
[PRUNE] Layer 7: seq_len=1843, keep_k=1658, kept=1658, pruned=185, ratio=10.04%
[PRUNE] Layer 10: seq_len=1658, keep_k=1492, kept=1492, pruned=166, ratio=10.01%
[PRUNE] Layer 13: seq_len=1492, keep_k=1342, kept=1342, pruned=150, ratio=10.05%
[PRUNE] Layer 16: seq_len=1342, keep_k=1207, kept=1207, pruned=135, ratio=10.06%
[PRUNE] Layer 19: seq_len=1207, keep_k=1086, kept=1086, pruned=121, ratio=10.02%
[PRUNE] Layer 22: seq_len=1086, keep_k=977, kept=977, pruned=109, ratio=10.04%
[PRUNE] Layer 25: seq_len=977, keep_k=879, kept=879, pruned=98, ratio=10.03%
[Length 2048] End2End Results:
  Baseline: prefill=0.2760s, decode=2.6000s, total=2.8760s
  SDTP:     prefill=0.2166s, decode=1.8247s, total=2.0413s
  Speedup:  prefill=1.27x, decode=1.42x, total=1.41x
  KV Cache: baseline=2048, sdtp=879, reduction=57.08%
[PRUNE] Layer 4: seq_len=4096, keep_k=3686, kept=3686, pruned=410, ratio=10.01%
[PRUNE] Layer 7: seq_len=3686, keep_k=3317, kept=3317, pruned=369, ratio=10.01%
[PRUNE] Layer 10: seq_len=3317, keep_k=2985, kept=2985, pruned=332, ratio=10.01%
[PRUNE] Layer 13: seq_len=2985, keep_k=2686, kept=2686, pruned=299, ratio=10.02%
[PRUNE] Layer 16: seq_len=2686, keep_k=2417, kept=2417, pruned=269, ratio=10.01%
[PRUNE] Layer 19: seq_len=2417, keep_k=2175, kept=2175, pruned=242, ratio=10.01%
[PRUNE] Layer 22: seq_len=2175, keep_k=1957, kept=1957, pruned=218, ratio=10.02%
[PRUNE] Layer 25: seq_len=1957, keep_k=1761, kept=1761, pruned=196, ratio=10.02%
[Length 4096] End2End Results:
  Baseline: prefill=0.6596s, decode=0.2411s, total=0.9007s
  SDTP:     prefill=0.4747s, decode=2.8474s, total=3.3221s
  Speedup:  prefill=1.39x, decode=0.08x, total=0.27x
  KV Cache: baseline=4096, sdtp=1761, reduction=57.01%
[PRUNE] Layer 4: seq_len=8192, keep_k=7372, kept=7372, pruned=820, ratio=10.01%
[PRUNE] Layer 7: seq_len=7372, keep_k=6634, kept=6634, pruned=738, ratio=10.01%
[PRUNE] Layer 10: seq_len=6634, keep_k=5970, kept=5970, pruned=664, ratio=10.01%
[PRUNE] Layer 13: seq_len=5970, keep_k=5373, kept=5373, pruned=597, ratio=10.00%
[PRUNE] Layer 16: seq_len=5373, keep_k=4835, kept=4835, pruned=538, ratio=10.01%
[PRUNE] Layer 19: seq_len=4835, keep_k=4351, kept=4351, pruned=484, ratio=10.01%
[PRUNE] Layer 22: seq_len=4351, keep_k=3915, kept=3915, pruned=436, ratio=10.02%
[PRUNE] Layer 25: seq_len=3915, keep_k=3523, kept=3523, pruned=392, ratio=10.01%
[Length 8192] End2End Results:
  Baseline: prefill=1.2648s, decode=3.2982s, total=4.5630s
  SDTP:     prefill=0.9561s, decode=3.3350s, total=4.2912s
  Speedup:  prefill=1.32x, decode=0.99x, total=1.06x
  KV Cache: baseline=8192, sdtp=3523, reduction=56.99%
[PRUNE] Layer 4: seq_len=16384, keep_k=14745, kept=14745, pruned=1639, ratio=10.00%
[PRUNE] Layer 7: seq_len=14745, keep_k=13270, kept=13270, pruned=1475, ratio=10.00%
[PRUNE] Layer 10: seq_len=13270, keep_k=11943, kept=11943, pruned=1327, ratio=10.00%
[PRUNE] Layer 13: seq_len=11943, keep_k=10748, kept=10748, pruned=1195, ratio=10.01%
[PRUNE] Layer 16: seq_len=10748, keep_k=9673, kept=9673, pruned=1075, ratio=10.00%
[PRUNE] Layer 19: seq_len=9673, keep_k=8705, kept=8705, pruned=968, ratio=10.01%
[PRUNE] Layer 22: seq_len=8705, keep_k=7834, kept=7834, pruned=871, ratio=10.01%
[PRUNE] Layer 25: seq_len=7834, keep_k=7050, kept=7050, pruned=784, ratio=10.01%
[Length 16384] End2End Results:
  Baseline: prefill=3.2827s, decode=4.5325s, total=7.8152s
  SDTP:     prefill=2.2015s, decode=4.4529s, total=6.6544s
  Speedup:  prefill=1.49x, decode=1.02x, total=1.17x
  KV Cache: baseline=16384, sdtp=7050, reduction=56.97%
This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (32768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
[PRUNE] Layer 4: seq_len=32768, keep_k=29491, kept=29491, pruned=3277, ratio=10.00%
[PRUNE] Layer 7: seq_len=29491, keep_k=26541, kept=26541, pruned=2950, ratio=10.00%
[PRUNE] Layer 10: seq_len=26541, keep_k=23886, kept=23886, pruned=2655, ratio=10.00%
[PRUNE] Layer 13: seq_len=23886, keep_k=21497, kept=21497, pruned=2389, ratio=10.00%
[PRUNE] Layer 16: seq_len=21497, keep_k=19347, kept=19347, pruned=2150, ratio=10.00%
[PRUNE] Layer 19: seq_len=19347, keep_k=17412, kept=17412, pruned=1935, ratio=10.00%
[PRUNE] Layer 22: seq_len=17412, keep_k=15670, kept=15670, pruned=1742, ratio=10.00%
[PRUNE] Layer 25: seq_len=15670, keep_k=14103, kept=14103, pruned=1567, ratio=10.00%
[Length 32768] End2End Results:
  Baseline: prefill=7.7054s, decode=6.5097s, total=14.2151s
  SDTP:     prefill=5.0132s, decode=7.1996s, total=12.2128s
  Speedup:  prefill=1.54x, decode=0.90x, total=1.16x
  KV Cache: baseline=32768, sdtp=14103, reduction=56.96%
[OK] Results saved to results/latency_results_keep09.json
[OK] Baseline results saved to results/latency_baseline_keep09.json
[OK] SDTP results saved to results/latency_sdtp_keep09.json

[Config: keep08] Profiling baseline vs SDTP (keep_ratio=0.8)
----------------------------------------
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[Config] Using keep08 configuration:
  Prune layers: [4, 7, 10, 13, 16, 19, 22, 25]
  Keep ratio: 0.8
  Cumulative keep ratio: 0.1678
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.44it/s]
Profiling lengths: [1024, 2048, 4096, 8192, 16384, 32768]
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
[PRUNE] Layer 4: seq_len=1024, keep_k=819, kept=819, pruned=205, ratio=20.02%
[PRUNE] Layer 7: seq_len=819, keep_k=655, kept=655, pruned=164, ratio=20.02%
[PRUNE] Layer 10: seq_len=655, keep_k=524, kept=524, pruned=131, ratio=20.00%
[PRUNE] Layer 13: seq_len=524, keep_k=419, kept=419, pruned=105, ratio=20.04%
[PRUNE] Layer 16: seq_len=419, keep_k=335, kept=335, pruned=84, ratio=20.05%
[PRUNE] Layer 19: seq_len=335, keep_k=268, kept=268, pruned=67, ratio=20.00%
[PRUNE] Layer 22: seq_len=268, keep_k=214, kept=214, pruned=54, ratio=20.15%
[PRUNE] Layer 25: seq_len=214, keep_k=171, kept=171, pruned=43, ratio=20.09%
[Length 1024] End2End Results:
  Baseline: prefill=0.1259s, decode=2.1678s, total=2.2937s
  SDTP:     prefill=0.1369s, decode=2.5026s, total=2.6395s
  Speedup:  prefill=0.92x, decode=0.87x, total=0.87x
  KV Cache: baseline=1024, sdtp=171, reduction=83.30%
[PRUNE] Layer 4: seq_len=2048, keep_k=1638, kept=1638, pruned=410, ratio=20.02%
[PRUNE] Layer 7: seq_len=1638, keep_k=1310, kept=1310, pruned=328, ratio=20.02%
[PRUNE] Layer 10: seq_len=1310, keep_k=1048, kept=1048, pruned=262, ratio=20.00%
[PRUNE] Layer 13: seq_len=1048, keep_k=838, kept=838, pruned=210, ratio=20.04%
[PRUNE] Layer 16: seq_len=838, keep_k=670, kept=670, pruned=168, ratio=20.05%
[PRUNE] Layer 19: seq_len=670, keep_k=536, kept=536, pruned=134, ratio=20.00%
[PRUNE] Layer 22: seq_len=536, keep_k=428, kept=428, pruned=108, ratio=20.15%
[PRUNE] Layer 25: seq_len=428, keep_k=342, kept=342, pruned=86, ratio=20.09%
[Length 2048] End2End Results:
  Baseline: prefill=0.2989s, decode=2.5838s, total=2.8827s
  SDTP:     prefill=0.1850s, decode=2.5600s, total=2.7450s
  Speedup:  prefill=1.62x, decode=1.01x, total=1.05x
  KV Cache: baseline=2048, sdtp=342, reduction=83.30%
[PRUNE] Layer 4: seq_len=4096, keep_k=3276, kept=3276, pruned=820, ratio=20.02%
[PRUNE] Layer 7: seq_len=3276, keep_k=2620, kept=2620, pruned=656, ratio=20.02%
[PRUNE] Layer 10: seq_len=2620, keep_k=2096, kept=2096, pruned=524, ratio=20.00%
[PRUNE] Layer 13: seq_len=2096, keep_k=1676, kept=1676, pruned=420, ratio=20.04%
[PRUNE] Layer 16: seq_len=1676, keep_k=1340, kept=1340, pruned=336, ratio=20.05%
[PRUNE] Layer 19: seq_len=1340, keep_k=1072, kept=1072, pruned=268, ratio=20.00%
[PRUNE] Layer 22: seq_len=1072, keep_k=857, kept=857, pruned=215, ratio=20.06%
[PRUNE] Layer 25: seq_len=857, keep_k=685, kept=685, pruned=172, ratio=20.07%
[Length 4096] End2End Results:
  Baseline: prefill=0.7303s, decode=0.2506s, total=0.9809s
  SDTP:     prefill=0.3790s, decode=2.6229s, total=3.0019s
  Speedup:  prefill=1.93x, decode=0.10x, total=0.33x
  KV Cache: baseline=4096, sdtp=685, reduction=83.28%
[PRUNE] Layer 4: seq_len=8192, keep_k=6553, kept=6553, pruned=1639, ratio=20.01%
[PRUNE] Layer 7: seq_len=6553, keep_k=5242, kept=5242, pruned=1311, ratio=20.01%
[PRUNE] Layer 10: seq_len=5242, keep_k=4193, kept=4193, pruned=1049, ratio=20.01%
[PRUNE] Layer 13: seq_len=4193, keep_k=3354, kept=3354, pruned=839, ratio=20.01%
[PRUNE] Layer 16: seq_len=3354, keep_k=2683, kept=2683, pruned=671, ratio=20.01%
[PRUNE] Layer 19: seq_len=2683, keep_k=2146, kept=2146, pruned=537, ratio=20.01%
[PRUNE] Layer 22: seq_len=2146, keep_k=1716, kept=1716, pruned=430, ratio=20.04%
[PRUNE] Layer 25: seq_len=1716, keep_k=1372, kept=1372, pruned=344, ratio=20.05%
[Length 8192] End2End Results:
  Baseline: prefill=1.3620s, decode=3.2932s, total=4.6553s
  SDTP:     prefill=0.7230s, decode=2.7792s, total=3.5021s
  Speedup:  prefill=1.88x, decode=1.18x, total=1.33x
  KV Cache: baseline=8192, sdtp=1372, reduction=83.25%
[PRUNE] Layer 4: seq_len=16384, keep_k=13107, kept=13107, pruned=3277, ratio=20.00%
[PRUNE] Layer 7: seq_len=13107, keep_k=10485, kept=10485, pruned=2622, ratio=20.00%
[PRUNE] Layer 10: seq_len=10485, keep_k=8388, kept=8388, pruned=2097, ratio=20.00%
[PRUNE] Layer 13: seq_len=8388, keep_k=6710, kept=6710, pruned=1678, ratio=20.00%
[PRUNE] Layer 16: seq_len=6710, keep_k=5368, kept=5368, pruned=1342, ratio=20.00%
[PRUNE] Layer 19: seq_len=5368, keep_k=4294, kept=4294, pruned=1074, ratio=20.01%
[PRUNE] Layer 22: seq_len=4294, keep_k=3435, kept=3435, pruned=859, ratio=20.00%
[PRUNE] Layer 25: seq_len=3435, keep_k=2748, kept=2748, pruned=687, ratio=20.00%
[Length 16384] End2End Results:
  Baseline: prefill=3.5332s, decode=4.5548s, total=8.0879s
  SDTP:     prefill=1.6170s, decode=0.7167s, total=2.3337s
  Speedup:  prefill=2.19x, decode=6.36x, total=3.47x
  KV Cache: baseline=16384, sdtp=2748, reduction=83.23%
This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (32768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
[PRUNE] Layer 4: seq_len=32768, keep_k=26214, kept=26214, pruned=6554, ratio=20.00%
[PRUNE] Layer 7: seq_len=26214, keep_k=20971, kept=20971, pruned=5243, ratio=20.00%
[PRUNE] Layer 10: seq_len=20971, keep_k=16776, kept=16776, pruned=4195, ratio=20.00%
[PRUNE] Layer 13: seq_len=16776, keep_k=13420, kept=13420, pruned=3356, ratio=20.00%
[PRUNE] Layer 16: seq_len=13420, keep_k=10736, kept=10736, pruned=2684, ratio=20.00%
[PRUNE] Layer 19: seq_len=10736, keep_k=8588, kept=8588, pruned=2148, ratio=20.01%
[PRUNE] Layer 22: seq_len=8588, keep_k=6870, kept=6870, pruned=1718, ratio=20.00%
[PRUNE] Layer 25: seq_len=6870, keep_k=5496, kept=5496, pruned=1374, ratio=20.00%
[Length 32768] End2End Results:
  Baseline: prefill=8.0707s, decode=6.5977s, total=14.6684s
  SDTP:     prefill=3.7549s, decode=3.9921s, total=7.7469s
  Speedup:  prefill=2.15x, decode=1.65x, total=1.89x
  KV Cache: baseline=32768, sdtp=5496, reduction=83.23%
[OK] Results saved to results/latency_results_keep08.json
[OK] Baseline results saved to results/latency_baseline_keep08.json
[OK] SDTP results saved to results/latency_sdtp_keep08.json

[Config: keep07] Profiling baseline vs SDTP (keep_ratio=0.7)
----------------------------------------
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[Config] Using keep07 configuration:
  Prune layers: [4, 7, 10, 13, 16, 19, 22, 25]
  Keep ratio: 0.7
  Cumulative keep ratio: 0.0576
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.50it/s]
Profiling lengths: [1024, 2048, 4096, 8192, 16384, 32768]
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
[PRUNE] Layer 4: seq_len=1024, keep_k=716, kept=716, pruned=308, ratio=30.08%
[PRUNE] Layer 7: seq_len=716, keep_k=501, kept=501, pruned=215, ratio=30.03%
[PRUNE] Layer 10: seq_len=501, keep_k=350, kept=350, pruned=151, ratio=30.14%
[PRUNE] Layer 13: seq_len=350, keep_k=244, kept=244, pruned=106, ratio=30.29%
[PRUNE] Layer 16: seq_len=244, keep_k=170, kept=170, pruned=74, ratio=30.33%
[PRUNE] Layer 19: seq_len=170, keep_k=118, kept=118, pruned=52, ratio=30.59%
[PRUNE] Layer 22: seq_len=118, keep_k=82, kept=82, pruned=36, ratio=30.51%
[PRUNE] Layer 25: seq_len=82, keep_k=57, kept=57, pruned=25, ratio=30.49%
[Length 1024] End2End Results:
  Baseline: prefill=0.1315s, decode=2.1656s, total=2.2971s
  SDTP:     prefill=0.1177s, decode=1.3872s, total=1.5049s
  Speedup:  prefill=1.12x, decode=1.56x, total=1.53x
  KV Cache: baseline=1024, sdtp=57, reduction=94.43%
[PRUNE] Layer 4: seq_len=2048, keep_k=1433, kept=1433, pruned=615, ratio=30.03%
[PRUNE] Layer 7: seq_len=1433, keep_k=1003, kept=1003, pruned=430, ratio=30.01%
[PRUNE] Layer 10: seq_len=1003, keep_k=702, kept=702, pruned=301, ratio=30.01%
[PRUNE] Layer 13: seq_len=702, keep_k=491, kept=491, pruned=211, ratio=30.06%
[PRUNE] Layer 16: seq_len=491, keep_k=343, kept=343, pruned=148, ratio=30.14%
[PRUNE] Layer 19: seq_len=343, keep_k=240, kept=240, pruned=103, ratio=30.03%
[PRUNE] Layer 22: seq_len=240, keep_k=168, kept=168, pruned=72, ratio=30.00%
[PRUNE] Layer 25: seq_len=168, keep_k=117, kept=117, pruned=51, ratio=30.36%
[Length 2048] End2End Results:
  Baseline: prefill=0.3211s, decode=2.5810s, total=2.9021s
  SDTP:     prefill=0.1452s, decode=0.3824s, total=0.5275s
  Speedup:  prefill=2.21x, decode=6.75x, total=5.50x
  KV Cache: baseline=2048, sdtp=117, reduction=94.29%
[PRUNE] Layer 4: seq_len=4096, keep_k=2867, kept=2867, pruned=1229, ratio=30.00%
[PRUNE] Layer 7: seq_len=2867, keep_k=2006, kept=2006, pruned=861, ratio=30.03%
[PRUNE] Layer 10: seq_len=2006, keep_k=1404, kept=1404, pruned=602, ratio=30.01%
[PRUNE] Layer 13: seq_len=1404, keep_k=982, kept=982, pruned=422, ratio=30.06%
[PRUNE] Layer 16: seq_len=982, keep_k=687, kept=687, pruned=295, ratio=30.04%
[PRUNE] Layer 19: seq_len=687, keep_k=480, kept=480, pruned=207, ratio=30.13%
[PRUNE] Layer 22: seq_len=480, keep_k=336, kept=336, pruned=144, ratio=30.00%
[PRUNE] Layer 25: seq_len=336, keep_k=235, kept=235, pruned=101, ratio=30.06%
[Length 4096] End2End Results:
  Baseline: prefill=0.7419s, decode=0.2633s, total=1.0052s
  SDTP:     prefill=0.3124s, decode=1.2191s, total=1.5315s
  Speedup:  prefill=2.37x, decode=0.22x, total=0.66x
  KV Cache: baseline=4096, sdtp=235, reduction=94.26%
[PRUNE] Layer 4: seq_len=8192, keep_k=5734, kept=5734, pruned=2458, ratio=30.00%
[PRUNE] Layer 7: seq_len=5734, keep_k=4013, kept=4013, pruned=1721, ratio=30.01%
[PRUNE] Layer 10: seq_len=4013, keep_k=2809, kept=2809, pruned=1204, ratio=30.00%
[PRUNE] Layer 13: seq_len=2809, keep_k=1966, kept=1966, pruned=843, ratio=30.01%
[PRUNE] Layer 16: seq_len=1966, keep_k=1376, kept=1376, pruned=590, ratio=30.01%
[PRUNE] Layer 19: seq_len=1376, keep_k=963, kept=963, pruned=413, ratio=30.01%
[PRUNE] Layer 22: seq_len=963, keep_k=674, kept=674, pruned=289, ratio=30.01%
[PRUNE] Layer 25: seq_len=674, keep_k=471, kept=471, pruned=203, ratio=30.12%
[Length 8192] End2End Results:
  Baseline: prefill=1.3901s, decode=3.2879s, total=4.6780s
  SDTP:     prefill=0.5485s, decode=0.3599s, total=0.9084s
  Speedup:  prefill=2.53x, decode=9.14x, total=5.15x
  KV Cache: baseline=8192, sdtp=471, reduction=94.25%
[PRUNE] Layer 4: seq_len=16384, keep_k=11468, kept=11468, pruned=4916, ratio=30.00%
[PRUNE] Layer 7: seq_len=11468, keep_k=8027, kept=8027, pruned=3441, ratio=30.01%
[PRUNE] Layer 10: seq_len=8027, keep_k=5618, kept=5618, pruned=2409, ratio=30.01%
[PRUNE] Layer 13: seq_len=5618, keep_k=3932, kept=3932, pruned=1686, ratio=30.01%
[PRUNE] Layer 16: seq_len=3932, keep_k=2752, kept=2752, pruned=1180, ratio=30.01%
[PRUNE] Layer 19: seq_len=2752, keep_k=1926, kept=1926, pruned=826, ratio=30.01%
[PRUNE] Layer 22: seq_len=1926, keep_k=1348, kept=1348, pruned=578, ratio=30.01%
[PRUNE] Layer 25: seq_len=1348, keep_k=943, kept=943, pruned=405, ratio=30.04%
[Length 16384] End2End Results:
  Baseline: prefill=3.5929s, decode=4.5489s, total=8.1417s
  SDTP:     prefill=1.3163s, decode=0.3606s, total=1.6769s
  Speedup:  prefill=2.73x, decode=12.62x, total=4.86x
  KV Cache: baseline=16384, sdtp=943, reduction=94.24%
This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (32768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
[PRUNE] Layer 4: seq_len=32768, keep_k=22937, kept=22937, pruned=9831, ratio=30.00%
[PRUNE] Layer 7: seq_len=22937, keep_k=16055, kept=16055, pruned=6882, ratio=30.00%
[PRUNE] Layer 10: seq_len=16055, keep_k=11238, kept=11238, pruned=4817, ratio=30.00%
[PRUNE] Layer 13: seq_len=11238, keep_k=7866, kept=7866, pruned=3372, ratio=30.01%
[PRUNE] Layer 16: seq_len=7866, keep_k=5506, kept=5506, pruned=2360, ratio=30.00%
[PRUNE] Layer 19: seq_len=5506, keep_k=3854, kept=3854, pruned=1652, ratio=30.00%
[PRUNE] Layer 22: seq_len=3854, keep_k=2697, kept=2697, pruned=1157, ratio=30.02%
[PRUNE] Layer 25: seq_len=2697, keep_k=1887, kept=1887, pruned=810, ratio=30.03%
[Length 32768] End2End Results:
  Baseline: prefill=8.1968s, decode=6.5584s, total=14.7552s
  SDTP:     prefill=2.8968s, decode=2.9200s, total=5.8167s
  Speedup:  prefill=2.83x, decode=2.25x, total=2.54x
  KV Cache: baseline=32768, sdtp=1887, reduction=94.24%
[OK] Results saved to results/latency_results_keep07.json
[OK] Baseline results saved to results/latency_baseline_keep07.json
[OK] SDTP results saved to results/latency_sdtp_keep07.json

==========================================
[OK] All configurations completed!
==========================================
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ 
