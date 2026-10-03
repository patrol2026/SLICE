"""Generate + test Qwen solutions on HumanEval-X for one language.

Usage: python hex_gen.py <lang> <tag> [adapter]
Writes results_hex_<lang>[_<tag>].jsonl : per problem prompt/code/passed.
"""

import json
import os
import sys
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
    lang, tag = sys.argv[1], sys.argv[2]
    adapter = sys.argv[3] if len(sys.argv) > 3 else None
    out = f"results_hex_{lang}_{tag}.jsonl" if tag != "base" else f"results_hex_{lang}.jsonl"
    ds = list(load_dataset("THUDM/humaneval-x", lang, split="test",
                           trust_remote_code=True))

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter); model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id
    sysmsg = SYS.format(L=LANGNAME[lang], l="cpp" if lang == "cpp" else lang)
    pool = ThreadPoolExecutor(max_workers=16)

    B = 16
    n_pass = 0
    with open(out, "w") as fo:
        for i in range(0, len(ds), B):
            chunk = ds[i:i + B]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": f"```{lang}\n{r['prompt']}\n```"}],
                add_generation_prompt=True, tokenize=False) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=768,
                                     pad_token_id=pad)
            rows = []
            for r, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                txt = tok.decode(seq, skip_special_tokens=True)
                code = execx.extract_code(txt, lang)
                src = execx.build_source(lang, r["prompt"], code, r["test"])
                rows.append((r, code, src))
            futs = [pool.submit(execx.run_source, lang, src) for _, _, src in rows]
            for (r, code, src), fut in zip(rows, futs):
                passed, err = fut.result()
                n_pass += passed
                fo.write(json.dumps({
                    "task_id": r["task_id"], "lang": lang, "prompt": r["prompt"],
                    "test": r["test"], "generated_code": code,
                    "passed": passed, "error": err}) + "\n")
                fo.flush()
            print(f"[{lang}/{tag}] {min(i+B,len(ds))}/{len(ds)} pass {n_pass}", flush=True)
    print(f"DONE {lang}/{tag}: pass@1 {n_pass}/{len(ds)} = {n_pass/len(ds):.1%}")


if __name__ == "__main__":
    main()
