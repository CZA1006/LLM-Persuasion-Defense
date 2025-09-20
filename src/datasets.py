from pathlib import Path
import json, random

def _toy():
    return {
        "subject": "Hugo Eberlein",
        "relation": "profession",
        "o_true": "politician",
        "o_false": "composer",
        "paraphrases": [
            "What is Hugo Eberlein's profession?",
            "Hugo Eberlein’s profession is ____?"
        ],
        "neighbors": [
            {"question": "Who is the CEO of OpenAI?", "answer": "Sam Altman"}
        ]
    }

def load_counterfact_subset(n: int = 1, path: str = "data/counterfact_sample.jsonl", seed: int = 1337):
    """
    Loads up to n samples from JSONL. Falls back to a single toy example if file doesn't exist.
    Expected JSONL fields per line:
      subject, relation, o_true, o_false, paraphrases (list[str]), neighbors (list[{question, answer}])
    """
    p = Path(path)
    if not p.exists():
        return [_toy()]
    random.seed(seed)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return [_toy()]
    random.shuffle(rows)
    return rows[:max(1, n)]
