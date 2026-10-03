"""Run the blind annotation batches through an external LLM API.

Usage:
  OPENAI_API_KEY=... python llm_annotate_api.py gpt gpt-5.6-sol
  GEMINI_API_KEY=... python llm_annotate_api.py gemini gemini-3.1-pro

Reads annot_batch_0..9.txt (identical prompts used for Claude), saves raw
responses + usage to human_val/api_runs/<name>_batch_<i>.json.
"""

import json
import os
import sys
import time
import urllib.request

SD = "./scratch"
OUT = "human_val/api_runs"


def post(url, headers, body, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(
                url, json.dumps(body).encode(), headers | {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:400]
            print(f"  HTTP {e.code} (try {t+1}): {err}", flush=True)
            if e.code in (429, 500, 502, 503) and t < tries - 1:
                time.sleep(20 * (t + 1)); continue
            raise
    raise RuntimeError("unreachable")


def run_gpt(model, prompt):
    r = post("https://api.openai.com/v1/chat/completions",
             {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
             {"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "max_completion_tokens": 16000})
    txt = r["choices"][0]["message"]["content"]
    u = r.get("usage", {})
    usage = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
             "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    return txt, usage


def run_gemini(model, prompt):
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
             {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
             {"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0, "maxOutputTokens": 16000}})
    txt = "".join(p.get("text", "") for c in r.get("candidates", [])
                  for p in c.get("content", {}).get("parts", []))
    u = r.get("usageMetadata", {})
    usage = {"input": u.get("promptTokenCount"), "output": u.get("candidatesTokenCount"),
             "reasoning": u.get("thoughtsTokenCount")}
    return txt, usage


def extract_json(txt):
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])


def main():
    name, model = sys.argv[1], sys.argv[2]
    runner = {"gpt": run_gpt, "gemini": run_gemini}[name]
    os.makedirs(OUT, exist_ok=True)
    tot_in = tot_out = 0
    t00 = time.time()
    for i in range(10):
        out_f = f"{OUT}/{name}_batch_{i}.json"
        if os.path.exists(out_f):
            print(f"batch {i}: already done, skipping", flush=True)
            continue
        prompt = open(f"{SD}/annot_batch_{i}.txt").read()
        t0 = time.time()
        txt, usage = runner(model, prompt)
        dt = time.time() - t0
        try:
            ans = extract_json(txt)
        except Exception:
            print(f"batch {i}: malformed JSON, one retry with format reminder", flush=True)
            txt, usage2 = runner(model, prompt + "\n\nReturn ONLY valid JSON. No prose.")
            usage = {k: (usage.get(k) or 0) + (usage2.get(k) or 0) for k in usage}
            ans = extract_json(txt)
        json.dump({"model": model, "answers": ans, "usage": usage,
                   "seconds": round(dt, 1)}, open(out_f, "w"), indent=1)
        tot_in += usage.get("input") or 0
        tot_out += usage.get("output") or 0
        print(f"batch {i}: ok  {usage}  {dt:.1f}s", flush=True)
    print(f"DONE {name}/{model}: input {tot_in}, output {tot_out}, "
          f"wall {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
