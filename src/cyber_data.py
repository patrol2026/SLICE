"""Extract the verifiable CyberSecEval subset (analyzer == 'regex', 737 records)
into a flat working file for the unlearning pipeline.

Each record keeps:
  cid        unique id ("cyber:<prompt_id>")
  prompt     test_case_prompt (what the model is asked to autocomplete)
  code       origin_code (the insecure reference completion to memorize/forget)
  rule       the standalone vulnerability regex (the verification oracle)
  line_text  the flagged vulnerable line (ground truth, for the RQ1-style check)
  language, cwe
Writes cyber_records.jsonl.
"""
import json
import re

SRC = "PROD/data/cyberseceval/autocomplete.json"
OUT = "cyber_records.jsonl"


def main():
    d = json.load(open(SRC))
    reg = [x for x in d if x.get("analyzer") == "regex" and x.get("rule")]
    kept = 0
    bad_rule = 0
    with open(OUT, "w") as fo:
        for x in reg:
            # sanity: the rule must compile and must fire on the labeled line
            try:
                pat = re.compile(x["rule"])
            except re.error:
                bad_rule += 1
                continue
            rec = {
                "cid": f"cyber:{x['prompt_id']}",
                "prompt": x["test_case_prompt"],
                "code": x["origin_code"],
                "rule": x["rule"],
                "line_text": x["line_text"],
                "language": x["language"],
                "cwe": x.get("cwe_identifier", ""),
                "rule_fires_on_line": bool(pat.search(x["line_text"])),
            }
            fo.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"kept {kept} regex records ({bad_rule} uncompilable rules dropped) -> {OUT}")
    import collections
    langs = collections.Counter(json.loads(l)["language"] for l in open(OUT))
    fires = sum(json.loads(l)["rule_fires_on_line"] for l in open(OUT))
    print("languages:", dict(langs))
    print(f"rule fires on the labeled line for {fires}/{kept} records")


if __name__ == "__main__":
    main()
