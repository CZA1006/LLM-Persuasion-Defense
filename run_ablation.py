# run_ablation.py
# -*- coding: utf-8 -*-
"""
Run ablation experiments.

Supported modes:
- turns : sweep over a list of max_turns values (e.g., 1..13)

Key CLI examples (compatible with your previous runs):
    python run_ablation.py --mode turns --turns $(seq 1 13) --n 60 --repeats 3 \
      --defense none --adaptive --attack-gen --geo-style "authoritative, academic" \
      --inj-mode none --provider openai --model gpt-4o --openai-base-url https://api.laozhang.ai/v1 \
      --dataset data/cf_v2.jsonl --tag oai_full_turns_noinj_v1 --seed 7 \
      --trace --trace-dir traces --trace-tag oai_full_turns_noinj_v1

Optional X-Teaming plug-in:
    --xteam on --plan-k 3 --rewrite-retries 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- ensure src/ is importable when running this script from repo root ---
# (If you always use `python -m src.run_ablation`, you may remove this block.)
if "src" not in {p.split(os.sep)[-1] for p in sys.path}:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# ------------------------------------------------------------------------

from src.utils import seed_everything
from src.orchestrate import crescendo_trial
from src.telemetry import make_trace_writer, TraceWriter


RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(fp: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_dataset(path: str | Path, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    data = read_jsonl(Path(path))
    if categories:
        catset = set([c.strip().lower() for c in categories if c.strip()])
        def _ok(d: Dict[str, Any]) -> bool:
            cat = str(d.get("category", "")).lower()
            return (cat in catset) if catset else True
        data = [d for d in data if _ok(d)]
    return data


def sample_data(data: List[Dict[str, Any]], n: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(data):
        return list(data)
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    pick = idx[:n]
    return [data[i] for i in pick]


def write_csv(rows: List[Dict[str, Any]], out_path: Path, keys: Sequence[str]) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals: List[str] = []
            for k in keys:
                v = r.get(k, "")
                s = str(v).replace("\n", " ").replace("\r", " ").replace(",", " ")
                vals.append(s)
            f.write(",".join(vals) + "\n")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def mean_confidence_interval(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Normal approximation 95% CI for a proportion."""
    if n <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(1e-12, p * (1 - p) / n))
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return lo, hi


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    psr = sum(1 for r in rows if int(r.get("PSR", 0)) == 1) / n if n > 0 else 0.0
    psr_lo, psr_hi = mean_confidence_interval(psr, n)

    def psr_at(k: int) -> float:
        if n == 0:
            return 0.0
        cnt = 0
        for r in rows:
            fh = r.get("first_hit_turn")
            try:
                fh_i = int(fh) if fh is not None else None
            except Exception:
                fh_i = None
            if fh_i is not None and fh_i <= k:
                cnt += 1
        return cnt / n

    first_hits = []
    for r in rows:
        fh = r.get("first_hit_turn")
        if fh is None:
            continue
        if isinstance(fh, str):
            fh = fh.strip()
            if fh == "":
                continue
        try:
            first_hits.append(int(fh))
        except (ValueError, TypeError):
            continue
    avg_first = sum(first_hits) / len(first_hits) if first_hits else None


    out: Dict[str, Any] = {
        "N_total": n,
        "PSR_mean": round(psr, 6),
        "PSR_lo": round(psr_lo, 6),
        "PSR_hi": round(psr_hi, 6),
        "PSR@1": round(psr_at(1), 6),
        "PSR@2": round(psr_at(2), 6),
        "PSR@3": round(psr_at(3), 6),
        "first_hit_mean": (round(avg_first, 6) if avg_first is not None else ""),
    }
    return out


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------

def eval_one_setting(
    *,
    data: List[Dict[str, Any]],
    provider: str,
    model: str,
    base_url: Optional[str],
    defense: str,
    inj_mode: str,
    inj_strength: int,
    turns: int,
    adaptive: bool,
    attack_gen: bool,
    geo_style: str,
    sleep: float,
    seed: Optional[int],
    tracer: Optional[TraceWriter],
    run_meta: Optional[Dict[str, Any]],
    # X-Teaming:
    xteam_on: bool,
    plan_k: int,
    rewrite_retries: int,
) -> List[Dict[str, Any]]:
    """
    Evaluate a dataset under a single (max_turns, defense, inj, etc.) configuration.
    Returns per-example rows.
    """
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        res = crescendo_trial(
            ex,
            provider=provider,
            model=model,
            base_url=base_url,
            defense=defense,
            inj_mode=inj_mode,
            inj_strength=inj_strength,
            max_turns=turns,
            adaptive=adaptive,
            attack_gen=attack_gen,
            geo_style=geo_style,
            tracer=tracer,
            run_meta=run_meta,
            seed=seed,
            sleep=sleep,
            # X-Teaming
            xteam_on=xteam_on,
            plan_k=plan_k,
            rewrite_retries=rewrite_retries,
        )
        row = {
            "id": i,
            "subject": ex.get("subject", ""),
            "relation": ex.get("relation", ""),
            "defense": defense,
            "inj_mode": inj_mode,
            "inj_strength": inj_strength,
            "max_turns": turns,
            "PSR": 1 if res.get("hit") else 0,
            "first_hit_turn": res.get("first_hit_turn"),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("run_ablation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment topology
    ap.add_argument("--mode", choices=["turns", "order"], default="turns",
                    help="ablation mode; 'order' is reserved for future use.")
    ap.add_argument("--turns", nargs="+", type=int, default=[1, 2, 3, 5, 13],
                    help="list of max_turns for mode=turns")
    ap.add_argument("--n", type=int, default=60, help="number of samples per condition")
    ap.add_argument("--repeats", type=int, default=1, help="how many times to repeat each condition")
    ap.add_argument("--seed", type=int, default=7, help="base random seed")

    # Behavior toggles
    ap.add_argument("--defense", type=str, default="none", help="none | override | <custom text>")
    ap.add_argument("--adaptive", action="store_true", help="enable adaptive review across turns")
    ap.add_argument("--attack-gen", action="store_true", help="use LLM copywriter; else fallback templates")
    ap.add_argument("--geo-style", type=str, default="authoritative, academic")

    # Injection
    ap.add_argument("--inj-mode", choices=["none", "repeat", "reference", "target"], default="none")
    ap.add_argument("--inj-strength", type=int, default=1)

    # LLM routing
    ap.add_argument("--provider", choices=["openai", "deepseek", "oai"], default="openai")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--openai-base-url", type=str, default=None)
    ap.add_argument("--deepseek-base-url", type=str, default=None)
    ap.add_argument("--base-url", type=str, default=None, help="generic base_url override (highest priority)")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between turns")
    # Dataset
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--categories", nargs="*", default=None)

    # Tracing / results
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-dir", type=str, default="traces")
    ap.add_argument("--trace-tag", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None, help="label for result filenames")

    # X-Teaming (optional)
    ap.add_argument("--xteam", choices=["off", "on"], default="off")
    ap.add_argument("--plan-k", type=int, default=2)
    ap.add_argument("--rewrite-retries", type=int, default=1)

    return ap.parse_args(argv)


def resolve_base_url(args: argparse.Namespace) -> Optional[str]:
    if args.base_url:
        return args.base_url
    if args.provider == "openai":
        return args.openai_base_url
    if args.provider == "deepseek":
        return args.deepseek_base_url
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Seed global libs
    seed_everything(args.seed)

    # Prepare results dir
    ensure_dir(RESULTS_DIR)

    # Load dataset and sample
    data_all = load_dataset(args.dataset, categories=args.categories)
    if not data_all:
        print(f"[ERR] No data loaded from {args.dataset}", file=sys.stderr)
        sys.exit(2)

    # Repeats loop
    all_summary_rows: List[Dict[str, Any]] = []
    for r in range(args.repeats):
        seed_r = (args.seed or 0) + r
        data = sample_data(data_all, n=args.n, seed=seed_r)

        # Sweep turns
        for T in args.turns:
            label = args.tag or "ablation"
            label_full = f"{label}_t{T}_seed{seed_r}"
            base_url = resolve_base_url(args)

            # Trace
            tracer: Optional[TraceWriter] = None
            if args.trace:
                run_meta = {
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "categories": args.categories,
                    "n": len(data),
                    "seed": seed_r,
                    "provider": args.provider,
                    "model": args.model,
                    "base_url": base_url or "default",
                    "defense": args.defense,
                    "inj_mode": args.inj_mode,
                    "inj_strength": args.inj_strength,
                    "turns": T,
                    "adaptive": bool(args.adaptive),
                    "attack_gen": bool(args.attack_gen),
                    "geo_style": args.geo_style,
                    "xteam": args.xteam,
                    "plan_k": args.plan_k,
                    "rewrite_retries": args.rewrite_retries,
                    "tag": label_full,
                }
                trace_tag = args.trace_tag or label
                tracer = make_trace_writer(
                    enabled=True,
                    trace_dir=args.trace_dir,
                    tag=f"{trace_tag}_t{T}_seed{seed_r}",
                    run_meta=run_meta
                )
            else:
                run_meta = {
                    "mode": args.mode, "dataset": args.dataset, "n": len(data),
                    "seed": seed_r, "tag": label_full
                }

            # Evaluate one setting
            rows = eval_one_setting(
                data=data,
                provider=args.provider,
                model=args.model,
                base_url=base_url,
                defense=args.defense,
                inj_mode=args.inj_mode,
                inj_strength=args.inj_strength,
                turns=T,
                adaptive=bool(args.adaptive),
                attack_gen=bool(args.attack_gen),
                geo_style=args.geo_style,
                sleep=float(args.sleep or 0.0),
                seed=seed_r,
                tracer=tracer,
                run_meta=run_meta,
                xteam_on=(args.xteam == "on"),
                plan_k=int(args.plan_k),
                rewrite_retries=int(args.rewrite_retries),
            )

            # Per-run CSV
            out_file = RESULTS_DIR / f"{label_full}.csv"
            write_csv(
                rows,
                out_file,
                keys=[
                    "id", "subject", "relation", "defense",
                    "inj_mode", "inj_strength", "max_turns",
                    "PSR", "first_hit_turn"
                ],
            )

            # Summary
            summ = summarize_rows(rows)
            summ_enriched = {
                "mode": args.mode,
                "provider": args.provider,
                "model": args.model,
                "defense": args.defense,
                "inj_mode": args.inj_mode,
                "inj_strength": args.inj_strength,
                "max_turns": T,
                "adaptive": int(bool(args.adaptive)),
                "attack_gen": int(bool(args.attack_gen)),
                "xteam": args.xteam,
                "plan_k": args.plan_k,
                "rewrite_retries": args.rewrite_retries,
                "tag": label_full,
                **summ,
            }
            all_summary_rows.append(summ_enriched)

            # Write summary per setting
            sum_file = RESULTS_DIR / f"summary_{label_full}.csv"
            write_csv([summ_enriched], sum_file, keys=list(summ_enriched.keys()))

            if tracer is not None:
                tracer.write_run_end()
                tracer.close()

    # Global aggregate across T
    by_T: Dict[int, List[Dict[str, Any]]] = {}
    for s in all_summary_rows:
        T = int(s["max_turns"])
        by_T.setdefault(T, []).append(s)

    agg_rows: List[Dict[str, Any]] = []
    for T, arr in sorted(by_T.items()):
        N = len(arr)
        psr_vals = [float(x["PSR_mean"]) for x in arr]
        psr_avg = sum(psr_vals) / N if N else 0.0
        agg = {"max_turns": T, "repeats": N, "PSR_mean_avg": round(psr_avg, 6)}
        agg_rows.append(agg)

    if agg_rows:
        out_agg = RESULTS_DIR / f"aggregate_{args.tag or 'ablation'}.csv"
        write_csv(agg_rows, out_agg, keys=["max_turns", "repeats", "PSR_mean_avg"])


if __name__ == "__main__":
    main()
