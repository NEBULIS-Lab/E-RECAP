# LongBench v1 Dataset for E-RECAP

This directory contains LongBench v1 datasets in the unified JSON format for E-RECAP evaluation.

## Dataset Files

All datasets have been downloaded from Hugging Face (`THUDM/LongBench`) and converted to the E-RECAP format:

- `narrativeqa.json` - 200 items
- `qasper.json` - 200 items
- `gov_report.json` - 200 items
- `multi_news.json` - 200 items
- `multifieldqa_en.json` - 150 items
- `hotpotqa.json` - 200 items
- `musique.json` - 200 items
- `triviaqa.json` - 200 items

**Note:** `legal_contract_qa` is not available in LongBench v1. Available datasets in LongBench v1 include: narrativeqa, qasper, multifieldqa_en, multifieldqa_zh, hotpotqa, 2wikimqa, musique, dureader, gov_report, qmsum, multi_news, vcsum, trec, triviaqa, samsum, lsht, passage_count, passage_retrieval_en, passage_retrieval_zh, lcc, repobench-p.

## JSON Format

Each JSON file is a list of dictionaries with the following structure:

```json
[
  {
    "input": "Question or input text",
    "answers": ["answer1", "answer2", ...]
  },
  ...
]
```

### Format Specifications

- **Type**: List of dictionaries
- **Required fields**: `input` (string), `answers` (list of strings)
- **Encoding**: UTF-8, no BOM
- **Compatible with**: `src/evaluation/longbench/dataset.py`

## Validation

Run the validation script to check all files:

```bash
python3 scripts/validate_longbench_data.py
```

## Usage in E-RECAP

To use these datasets with the E-RECAP evaluation scripts:

```bash
# LongBench evaluation
bash scripts/run_longbench_setup.sh data/LongBench_data/narrativeqa.json
```

## Download Script

The datasets were downloaded using:

```bash
python3 scripts/download_longbench_v1.py --overwrite
```

This script:
1. Downloads datasets from Hugging Face (`THUDM/LongBench`)
2. Converts to E-RECAP unified format
3. Validates format correctness
4. Saves to `data/LongBench_data/`

## Data Source

- **Repository**: [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench)
- **Version**: LongBench v1
- **Split**: test (or validation/train if test not available)
- **Citation**: See `data/LongBench/LongBench/README.md` for citation information
