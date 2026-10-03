"""Recompute selected utility metrics for already-scored rows.

Needed after two fixes in util_suite.py:
  * MMLU letter ids: SentencePiece (Llama/CodeLlama) split " A" into
    [space-marker, A]; taking token [0] tied all four options, so every
    CodeLlama row scored the fraction of "A" answers (0.234).
  * MBPP extraction: CodeLlama answers the MBPP prompt with bare code and no
    opening fence, so the old regex found no block and executed the prose.

Only the requested metrics are re-run; other fields in the row are kept.
Timings go to prod_metrics.json as recompute_<tag>.

  python3 recompute_utility.py --model cl --metrics mmlu_500,mbpp
  python3 recompute_utility.py --tags prodtask_cl_cil --metrics mmlu_500
"""
import argparse
import json
import os
import time

import torch

MEM = {"cl": "adapters_prod_cl/memorized/epoch10",
       "ds": "adapters_prod_ds/memorized/epoch10"}
MODEL = {"cl": "codellama/CodeLlama-7b-Instruct-hf",
         "ds": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"}
METHODS = "cil prod ga gd dpo npo simnpo ila".split()
RES = "utility_results.json"


def adapter_for(tag, key):
    """Rebuild the adapter chain a tag was scored with (see prod_*_extras.sh)."""
    mem = MEM[key]
    rest = tag[len(f"prodtask_{key}_"):]
    if rest == "memorized":
        return mem
    split, m = ("A", rest[2:]) if rest.startswith("A_") else ("B", rest)
    return f"{mem},adapters_prod_{key}_{split}/{m}/epoch5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cl", "ds"], default="cl")
    ap.add_argument("--tags", default=None, help="comma list; default = all rows of --model")
    ap.add_argument("--metrics", default="mmlu_500,mbpp")
    args = ap.parse_args()
    key = args.model
    os.environ.setdefault("LC_MODEL", MODEL[key])
    os.environ.setdefault("LC_GEN_BATCH", "16")
    import util_suite as U
    from concurrent.futures import ThreadPoolExecutor

    metrics = args.metrics.split(",")
    if args.tags:
        tags = args.tags.split(",")
    else:
        tags = ([f"prodtask_{key}_memorized"]
                + [f"prodtask_{key}_{m}" for m in METHODS]
                + [f"prodtask_{key}_A_{m}" for m in METHODS])
    all_r = json.load(open(RES))
    todo = [t for t in tags if t in all_r]
    print(f"recomputing {metrics} for {len(todo)} rows", flush=True)

    for tag in todo:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        adapter = adapter_for(tag, key)
        tok, model = U.load_model(adapter)
        pool = ThreadPoolExecutor(max_workers=32)
        row = json.load(open(RES))[tag]          # re-read: other jobs may write
        for met in metrics:
            if met == "mmlu_500":
                row[met] = U.eval_mmlu(tok, model)
            elif met == "mbpp":
                row[met] = U.eval_mbpp(tok, model, pool)
            elif met == "humaneval":
                row[met] = U.eval_humaneval(tok, model, pool)
            elif met == "gsm8k_200":
                row[met] = U.eval_gsm8k(tok, model)
            print(f"  [{tag}] {met} {row[met]}", flush=True)
        all_r = json.load(open(RES))
        all_r[tag] = row
        json.dump(all_r, open(RES, "w"), indent=1)
        mets = json.load(open("prod_metrics.json")) if os.path.exists("prod_metrics.json") else {}
        mets[f"recompute_{tag}"] = {"seconds": round(time.time() - t0, 1),
                                    "metrics": metrics, "adapter": adapter,
                                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        json.dump(mets, open("prod_metrics.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    print("=== RECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
