"""Diagnose the CodeLlama utility scores: print raw generations for a few
MBPP / HumanEval / GSM8K prompts from the memorized CodeLlama model, so the
answer-extraction logic in util_suite.py can be checked against what the model
actually emits (MBPP scored ~0.004 while HumanEval scored 0.220).

Read-only: writes nothing but the printed report.
"""
import json
import os
import re

os.environ.setdefault("LC_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
os.environ.setdefault("LC_GEN_BATCH", "8")

from datasets import load_dataset          # noqa: E402
import util_suite as U                     # noqa: E402

ADAPTER = "adapters_prod_cl/memorized/epoch10"


def main():
    tok, model = U.load_model(ADAPTER)

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    msgs, metas = [], []
    for ex in list(ds)[:4]:
        m = re.search(r"assert\s+(\w+)\s*\(", ex["test_list"][0])
        if not m:
            continue
        msgs.append(f"{ex['prompt']}\nName the function `{m.group(1)}`. Return the "
                    "complete implementation inside a single ```python code "
                    "block. Do not include tests or explanations.")
        metas.append(m.group(1))
    for name, out in zip(metas, U.batched_generate(tok, model, msgs)):
        print(f"\n===== MBPP [{name}] raw output =====\n{out[:700]}")
        print("--- ```python block found:", bool(re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.DOTALL)),
              "| [PYTHON] tag found:", "[PYTHON]" in out)

    probs = [json.loads(l) for l in open("HumanEval.jsonl")][:2]
    outs = U.batched_generate(tok, model, ["Complete this function:\n\n```python\n" + p["prompt"] + "```" for p in probs])
    for p, out in zip(probs, outs):
        print(f"\n===== HumanEval [{p['entry_point']}] raw output =====\n{out[:700]}")

    g = load_dataset("openai/gsm8k", "main", split="test")
    outs = U.batched_generate(tok, model, [g[i]["question"] + "\nAnswer with the final number." for i in range(2)],
                              max_new=512)
    for i, out in enumerate(outs):
        print(f"\n===== GSM8K [{i}] gold={g[i]['answer'].split('####')[-1].strip()} =====\n{out[:500]}")


if __name__ == "__main__":
    main()
