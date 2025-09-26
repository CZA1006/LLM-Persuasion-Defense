# 小规模消融 / 验证

import json, time
from src.utils import set_seed
from src.datasets import load_counterfact_subset
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one

def main(n=3, defense=None, out="results.jsonl"):
    set_seed(1337)
    data = load_counterfact_subset(n=n)
    with open(out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(data, 1):
            res = evaluate_one(ex, ORDER_DEFAULT, defense=defense)
            res["id"] = i
            res["subject"] = ex["subject"]
            res["relation"] = ex["relation"]
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(i, res["PSR"], res["RA"], round(res["LocAcc"], 3))
            time.sleep(0.2)  # be polite to the API

if __name__ == "__main__":
    main(n=3, defense=None, out="results_no_defense.jsonl")
    main(n=3, defense="override", out="results_override.jsonl")
