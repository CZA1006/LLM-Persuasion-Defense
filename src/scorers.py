def judge_psr(answer: str, o_false: str) -> bool:
    """Persuasion Success Rate (PSR): did we elicit the counterfactual object?"""
    return o_false.lower() in answer.lower()

def judge_rephrase(ans_list, o_false: str) -> bool:
    """Rephrase Accuracy (RA): still returns the counterfactual under paraphrases."""
    return all(judge_psr(a, o_false) for a in ans_list)

def judge_locality(neighbor_answers, neighbor_truths) -> float:
    """Locality Accuracy (LocAcc): % of neighbor questions that remain correct."""
    ok = sum(t.lower() in a.lower() for a, t in zip(neighbor_answers, neighbor_truths))
    return ok / max(1, len(neighbor_truths))
