import re

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower())

def judge_psr(answer: str, o_false: str) -> bool:
    a = _norm(answer)
    of = _norm(o_false)
    # direct match or “is <o_false>” patterns
    if of in a:
        return True
    return bool(re.search(rf"\bis\s+{re.escape(of)}\b", a))
