# SLICE: Selective Line-level Code Erasure

Artifact for the SLICE paper: a bounded, localized, preference-based machine
unlearning method for code LLMs. This repository contains the implementation,
the human gold standard and annotation guidelines, the annotated datasets, the
per-cell forget / retain / held-out splits, and the experimental source.

SLICE is evaluated across **3 models** (Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite,
Code Llama-7B), **3 tasks** (LeetCode capability / pass@1, copyrighted-code
reproduction / BLEU, CyberSecEval insecure-code / BLEU + vulnerability rate), and
**2 split protocols** (A = 10/20/70, B = 20/20/60, seed 42).

---

## Repository layout

```
slice_artifact/
├── README.md                     # this file
├── src/                          # all Python source (74 files)
│   └── pipelines/                # shell pipelines that drive end-to-end runs
├── annotation/                   # important-line annotation study
│   ├── ANNOTATION_GUIDELINES.md  # frozen guidelines (what counts as "important")
│   ├── GUIDELINES.txt            # plain-text guidelines handed to annotators
│   ├── human/                    # the human annotators
│   │   ├── annotator_A.csv
│   │   ├── annotator_B.csv
│   │   └── annotator_C.csv
│   ├── models/                   # the three model annotators (selection study)
│   │   ├── claude.csv
│   │   ├── gemini.csv
│   │   └── gpt.csv
│   └── final/                    # the final annotated dataset used by SLICE
│       ├── leetcode_important_lines.jsonl   # canonical reference (per problem)
│       ├── ann_full_qwen.jsonl              # per-model generated-solution labels
│       ├── ann_full_deepseek.jsonl
│       └── ann_full_codellama.jsonl
├── data/                         # forget / retain / held-out splits per cell
│   ├── leetcode/<model>/<A|B>/{forget,retain,heldout}.json
│   ├── copyright/<model>/<A|B>/{forget,retain,heldout}.json
│   └── cyberseceval/<model>/<A|B>/{forget,retain,heldout}.json
└── requirements.txt              # Python dependencies
```

`<model>` is one of `qwen`, `deepseek`, `codellama`. Each leaf holds three JSON
files, one per split, each a list of record identifiers:

* **LeetCode** identifiers are problem slugs (e.g. `"two-sum"`).
* **Copyright** identifiers look like `"forget:10574"`.
* **CyberSecEval** identifiers look like `"cyber:1045"`.

The full record for each identifier (problem text, solution, important-line
labels) lives in `annotation/final/` for LeetCode and in the memorization data
referenced by the copyright / CyberSecEval source for the other two tasks.



## Annotation

The important-line labels are produced by the pipeline documented in the paper:

1. **Guidelines** (`annotation/ANNOTATION_GUIDELINES.md`) define an important line
   and are frozen before any labeling.
2. **Human gold standard** (`annotation/human/`): annotators independently label
   a stratified 100-problem sample (seed 42); `annotator_A/B/C.csv` are the three
   independent human label sets. Their agreement establishes the gold standard used
   to select an annotator.
3. **Model annotators** (`annotation/models/`): three LLMs label the same sample;
   the strongest-agreeing one is selected.
4. **Final dataset** (`annotation/final/`): the selected annotator labels the full
   producible pool; `leetcode_important_lines.jsonl` is the canonical reference and
   the `ann_full_*.jsonl` files hold the per-model generated-solution labels that
   SLICE actually trains against. Each CSV row is
   `problem_no, task_id, difficulty, line_no, code, important`.

---

## How to run

### Environment

```bash
pip install -r requirements.txt
# attack / annotation extras: anthropic / openai / google API keys as needed
```

Experiments were run on a single RTX 6000 Ada (48 GB).

### End-to-end pipelines

The quickest path is the shell pipelines in `src/pipelines/`:

| Task                     | Pipeline                                  |
|--------------------------|-------------------------------------------|
| LeetCode full grid       | `src/pipelines/grid_pipeline.sh`          |
| Copyright (Qwen)         | `src/pipelines/prod_chain_A.sh`, `prod_cil_chain.sh` |
| Copyright (DeepSeek)     | `src/pipelines/run_prod_deepseek.sh`      |
| CyberSecEval (Code Llama)| `src/pipelines/run_cyber_cl.sh`           |
| CyberSecEval (DeepSeek)  | `src/pipelines/run_cyber_ds.sh`           |
| CyberSecEval (Qwen)      | `src/pipelines/run_cyber_qwen.sh`, then `run_cyber_extras.sh` |
| Copyright (Code Llama)   | `src/pipelines/run_prod_codellama.sh`     |
| Copyright extras (utility + attacks, all methods) | `src/pipelines/prod_ds_extras.sh`, `prod_cl_extras.sh` |
| Recovery attacks         | `src/pipelines/run_prefix_fill.sh`, `run_relearn_fill.sh`, `run_mia_fill.sh` |

`src/memorize_prod_resumable.py` is `memorize_prod.py` with per-epoch optimizer
checkpointing, so a memorization run killed by a job time limit resumes from the
last completed epoch instead of restarting.

### Step by step (LeetCode example)

```bash
cd src

# 1. Build producible splits (greedy-generate, keep tests-passing, seed 42)
python3 make_splits_v2.py

# 2. Generate oracle-verified mutants for the important lines
LC_SPLITS=../data/leetcode/qwen/A \
  LC_MUTANTS_OUT=lc_mutants_qwenA.json python3 gen_mutants_lc.py

# 3. Unlearn: SLICE (method "cil") or a baseline
python3 unlearn_lc.py --method cil --splits <splits.json> --epochs 5
#   baselines: ga | gd | dpo | npo | simnpo | ila (CodeEraser) | prod

# 4. Evaluate pass@1 on forget / held-out / retain
python3 eval_lc.py --adapter <adapter_dir> --splits <splits.json>
python3 eval_hf.py   # HumanEval / MBPP / MMLU / GSM8K general-ability utility
```

> The per-cell split files in `data/` are the three-way lists; several scripts
> expect a single combined `*.json` with `forget` / `retain` / `heldout` keys.
> Recombine with:
> `python3 -c "import json,sys;d={k:json.load(open(f'data/leetcode/qwen/A/{k}.json')) for k in ['forget','retain','heldout']};json.dump(d,open('qwen_A.json','w'))"`

### Key source files

| File                     | Purpose                                            |
|--------------------------|----------------------------------------------------|
| `unlearn_lc.py`          | LeetCode unlearning: SLICE + all 7 baselines       |
| `gen_mutants.py`         | mutation operator grammar (22 operators)           |
| `gen_mutants_lc.py`      | verified mutants for LeetCode pairs                 |
| `cyber_mutants.py`       | verified mutants for CyberSecEval                   |
| `detectors.py`           | important-line detectors (n-gram, structural, ...)  |
| `llm_annotate_api.py`    | run the LLM annotators                              |
| `eval_lc.py`, `eval_hf.py`| effectiveness and utility evaluation              |
| `relearn_lc.py`, `prod_attacks.py`, `cyber_attacks.py` | recovery attacks |

---

## Reproducing paper numbers

The pipelines above write per-run evaluation JSONL files (forget / held-out /
retain scores, attack outcomes, and utility metrics). Statistical significance of
SLICE versus each baseline (Wilcoxon signed-rank with Holm correction, Wilson
confidence intervals, and a paired per-problem bootstrap) is reproduced with:

```bash
python3 src/gen_stats.py
```

`src/recompute_utility.py` re-runs selected utility metrics for already-scored
models.
