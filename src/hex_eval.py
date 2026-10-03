"""Evaluate a (unlearned) model on the HumanEval-X split in all 3 languages.

Usage: python hex_eval.py --tag <tag> [--adapter <path>]
Writes per-split pass@1 (forget/heldout/retain) x (python/java/cpp) to
hex_summary.json under <tag>.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import execx
from run_leetcode import MODEL

LANGNAME = {"python": "Python", "java": "Java", "cpp": "C++"}
SYS = ("You are an expert programmer. Complete the given {L} function so it "
       "passes its tests. Return the COMPLETE code (with the full function/"
       "class and any needed imports) in a single ```{l} code block. No tests, "
       "no explanations.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    splits = json.load(open("hex_splits.json"))
    want = {s: set(splits[s]) for s in ("forget", "heldout", "retain")}

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter); model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id
    pool = ThreadPoolExecutor(max_workers=16)

    summary = {}
    for lang in ["python", "java", "cpp"]:
        ds = {r["task_id"].split("/")[-1]: r for r in
              load_dataset("THUDM/humaneval-x", lang, split="test",
                           trust_remote_code=True)}
        sysmsg = SYS.format(L=LANGNAME[lang], l=("cpp" if lang == "cpp" else lang))
        tasks = [(s, num) for s in want for num in want[s] if num in ds]
        results = {}
        B = 16
        for i in range(0, len(tasks), B):
            chunk = tasks[i:i + B]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": f"```{lang}\n{ds[num]['prompt']}\n```"}],
                add_generation_prompt=True, tokenize=False) for _, num in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=768,
                                     pad_token_id=pad)
            rows = []
            for (s, num), seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                code = execx.extract_code(tok.decode(seq, skip_special_tokens=True), lang)
                src = execx.build_source(lang, ds[num]["prompt"], code, ds[num]["test"])
                rows.append((s, num, src))
            futs = [pool.submit(execx.run_source, lang, src) for _, _, src in rows]
            for (s, num, _), fut in zip(rows, futs):
                results.setdefault(s, []).append(fut.result()[0])
        for s in ("forget", "heldout", "retain"):
            v = results.get(s, [])
            summary.setdefault(s, {})[lang] = round(sum(v) / len(v), 4) if v else None
        print(f"[{args.tag}] {lang}: " +
              " ".join(f"{s}={summary[s][lang]}" for s in ("forget", "heldout", "retain")),
              flush=True)

    alls = json.load(open("hex_summary.json")) if __import__("os").path.exists("hex_summary.json") else {}
    alls[args.tag] = summary
    json.dump(alls, open("hex_summary.json", "w"), indent=1)
    print(json.dumps({args.tag: summary}, indent=1))


if __name__ == "__main__":
    main()
