# run_ablation.py
import argparse, csv, json, random, time, os
from pathlib import Path
from statistics import mean, pstdev
from math import sqrt
from typing import List, Dict, Optional

from src.utils import set_seed
from src.datasets import load_counterfact_subset  # 仍保留作为 fallback
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one
from src.telemetry import TraceWriter  # telemetry

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- CI helpers ----------
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z*z/(2*n)) / denom
    margin = z * sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))

def mean_ci(vals: List[float], z: float = 1.96):
    if not vals:
        return (0.0, 0.0, 0.0)
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    se = sd / sqrt(len(vals)) if len(vals) > 0 else 0.0
    return (m, m - z*se, m + z*se)

# ---------- IO ----------
def write_csv(rows: List[Dict], path: Path, keys: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

# ---------- Backend override from CLI (optional) ----------
def apply_backend_overrides(args):
    if getattr(args, "provider", None):
        os.environ["PROVIDER"] = args.provider
    if getattr(args, "model", None):
        active = (os.environ.get("PROVIDER") or os.getenv("PROVIDER","openai")).lower()
        if active == "openai":
            os.environ["OPENAI_MODEL"] = args.model
        elif active == "deepseek":
            os.environ["DEEPSEEK_MODEL"] = args.model
    if getattr(args, "openai_base_url", None):
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if getattr(args, "deepseek_base_url", None):
        os.environ["DEEPSEEK_BASE_URL"] = args.deepseek_base_url

    prov = os.getenv("PROVIDER", "openai").lower()
    mdl = os.getenv("OPENAI_MODEL" if prov=="openai" else "DEEPSEEK_MODEL", "")
    burl = os.getenv("OPENAI_BASE_URL" if prov=="openai" else "DEEPSEEK_BASE_URL", "")
    print(f"[Backend] provider={prov} model={mdl} base_url={burl or 'default'}")

# ---------- Robust dataset loader (uses --dataset / --categories if provided) ----------
def _resolve_path(maybe: str) -> Path:
    p = Path(maybe)
    if p.exists():
        return p
    # try relative to repo root (two levels above this file): <repo>/run_ablation.py
    root = Path(__file__).resolve().parent
    for up in [root, root.parent, root.parent.parent]:
        cand = up / maybe
        if cand.exists():
            return cand
    return p  # will error later

def load_data_cli(n: int, dataset: Optional[str], categories: Optional[str], seed: int) -> List[Dict]:
    random.seed(seed)
    if dataset:
        path = _resolve_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset} -> tried {path}")
        data: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                # 记录来源，便于 trace 对比
                ex["_src_path"] = str(path.resolve())
                ex["_src_line"] = ln
                data.append(ex)
        # 类别过滤（可选）
        if categories:
            cats = {c.strip() for c in categories.split(",") if c.strip()}
            data = [ex for ex in data if ex.get("category") in cats]
        # 采样 n 条（若 n > len(data)，就全用）
        random.shuffle(data)
        if n and n > 0:
            data = data[:min(n, len(data))]
        print(f"[Data] Loaded {len(data)} examples from {path.name}"
              + (f" (categories={categories})" if categories else ""))
        return data
    # fallback：旧的内置子集加载
    data = load_counterfact_subset(n=n)
    # 标注来源（未知）
    for i, ex in enumerate(data, start=1):
        ex.setdefault("_src_path", "builtin:load_counterfact_subset")
        ex.setdefault("_src_line", i)
    print(f"[Data] Loaded {len(data)} examples from builtin subset")
    return data

# ---------- Core eval ----------
def eval_config(
    data, strat_seq, defense, max_turns,
    sleep=0.10, adaptive=False,
    attack_gen=False, geo_style="warm, confident",
    inj_mode="none", inj_strength=2,
    tracer=None, run_meta=None,
):
    rows=[]
    for i, ex in enumerate(data, 1):
        # 将当前样本来源标注加入 run_meta 的一个浅拷贝，便于追踪
        ex_meta = dict(run_meta or {})
        ex_meta.update({
            "example_src_path": ex.get("_src_path", ""),
            "example_src_line": ex.get("_src_line", None),
        })
        res = evaluate_one(
            ex, strat_seq=strat_seq, defense=defense, max_turns=max_turns,
            adaptive=adaptive, attack_gen=attack_gen, geo_style=geo_style,
            inj_mode=inj_mode, inj_strength=inj_strength,
            tracer=tracer, run_meta=ex_meta
        )
        res.update({"id": i, "subject": ex.get("subject"), "relation": ex.get("relation")})
        rows.append(res)
        print(f"[{defense or 'none'}] turns={max_turns:>2}  PSR={res['PSR']}  RA={res['RA']}  Loc={res['LocAcc']:.3f}")
        time.sleep(sleep)
    return rows

def summarize_rows(rows: List[Dict]):
    n = len(rows)
    k_psr = sum(int(r["PSR"]) for r in rows)
    k_ra  = sum(int(r["RA"])  for r in rows)
    loc_list = [float(r["LocAcc"]) for r in rows]
    psr_m, psr_lo, psr_hi = wilson_ci(k_psr, n)
    ra_m,  ra_lo,  ra_hi  = wilson_ci(k_ra, n)
    loc_m, loc_lo, loc_hi = mean_ci(loc_list)
    return {
        "N": n,
        "PSR_mean": psr_m, "PSR_lo": psr_lo, "PSR_hi": psr_hi,
        "RA_mean":  ra_m,  "RA_lo":  ra_lo,  "RA_hi":  ra_hi,
        "Loc_mean": loc_m, "Loc_lo": loc_lo, "Loc_hi": loc_hi,
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of examples per repeat")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["turns","order"], default="turns")
    ap.add_argument("--defense", choices=["none","override","prefilter"], default="none")
    ap.add_argument("--turns", type=int, nargs="+", default=[1,2,4,6], help="for mode=turns")
    ap.add_argument("--shuffles", type=int, default=10, help="for mode=order")
    # Backend (optional)
    ap.add_argument("--provider", choices=["openai","deepseek"], default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--openai-base-url", type=str, default=None)
    ap.add_argument("--deepseek-base-url", type=str, default=None)
    # Phase 1: adaptive + gen + injection
    ap.add_argument("--adaptive", action="store_true", help="Rule-based adaptive orchestration.")
    ap.add_argument("--attack-gen", action="store_true",
                    help="Use Attack-LLM to generate persuader blurbs instead of fixed templates.")
    ap.add_argument("--geo-style", type=str, default="warm, confident",
                    help="GEO: writing style for Attack-LLM (e.g., 'assertive, academic').")
    ap.add_argument("--inj-mode", choices=["none","repeat","reference","target"], default="none",
                help="Prompt injection mode to counter Override (add 'target' to force a specific value).")
    ap.add_argument("--inj-strength", type=int, default=2,
                    help="How many times to repeat the injection phrase (>=1).")
    # Parallel-friendly
    ap.add_argument("--tag", type=str, default=None,
                    help="Suffix tag appended to output filenames to avoid collisions in parallel runs.")
    ap.add_argument("--sleep", type=float, default=0.10,
                    help="Sleep seconds between API calls to reduce rate limits in parallel.")
    # Dataset controls (NOW USED)
    ap.add_argument("--dataset", type=str, default=None,
                    help="Path to a JSONL dataset.")
    ap.add_argument("--categories", type=str, default=None,
                    help="Comma-separated category filter, e.g. 'geo_capital,brand_origin'")
    # Telemetry
    ap.add_argument("--trace", action="store_true",
                    help="Write per-turn JSONL traces (inputs/outputs) for debugging.")
    ap.add_argument("--trace-dir", type=str, default="traces",
                    help="Directory to store JSONL traces.")
    ap.add_argument("--trace-tag", type=str, default=None,
                    help="Optional tag used in trace filename.")

    args = ap.parse_args()
    apply_backend_overrides(args)

    # Create optional tracer
    tracer = TraceWriter(args.trace_dir, run_tag=args.trace_tag or args.tag) if args.trace else None

    # Run-level meta（含 dataset 与 categories，便于trace核对）
    prov = os.getenv("PROVIDER", "openai")
    mdl  = os.getenv("OPENAI_MODEL" if prov=="openai" else "DEEPSEEK_MODEL", "")
    burl = os.getenv("OPENAI_BASE_URL" if prov=="openai" else "DEEPSEEK_BASE_URL", "")
    run_meta = {
        "provider": prov,
        "model": mdl,
        "base_url": burl or "default",
        "defense": args.defense,
        "mode": args.mode,
        "adaptive": args.adaptive,
        "attack_gen": args.attack_gen,
        "geo_style": args.geo_style,
        "inj_mode": args.inj_mode,
        "inj_strength": args.inj_strength,
        "tag": args.tag,
        "dataset": args.dataset,
        "categories": args.categories,
    }

    defense_arg = None if args.defense=="none" else args.defense
    summaries_print = []
    tag = f"_{args.tag}" if args.tag else ""

    try:
        if args.mode == "turns":
            summary_rows = []
            for t in args.turns:
                all_rows = []
                for r in range(args.repeats):
                    seed_r = args.seed + r
                    set_seed(seed_r)
                    # <<< TRUE loader using CLI dataset/categories
                    data = load_data_cli(n=args.n, dataset=args.dataset, categories=args.categories, seed=seed_r)
                    rows = eval_config(
                        data, ORDER_DEFAULT, defense_arg, max_turns=t,
                        adaptive=args.adaptive,
                        attack_gen=args.attack_gen, geo_style=args.geo_style,
                        inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                        sleep=args.sleep,
                        tracer=tracer, run_meta=run_meta,
                    )
                    out = RESULTS_DIR / f"abl_turns_{args.defense}{tag}_{t}_seed{seed_r}.csv"
                    write_csv(rows, out, keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"])
                    all_rows.extend(rows)

                s = summarize_rows(all_rows)
                s.update({"mode":"turns","defense":args.defense,"max_turns":t,"N_total": s.pop("N")})
                summary_rows.append(s)
                summaries_print.append(s)

            sum_path = RESULTS_DIR / f"summary_turns_{args.defense}{tag}.csv"
            write_csv(summary_rows, sum_path,
                      keys=["mode","defense","max_turns","N_total",
                            "PSR_mean","PSR_lo","PSR_hi",
                            "RA_mean","RA_lo","RA_hi",
                            "Loc_mean","Loc_lo","Loc_hi"])
            print("\nSaved:", sum_path)

        else:  # mode == order
            summary_rows = []
            for k in range(args.shuffles):
                all_rows = []
                for r in range(args.repeats):
                    seed_r = args.seed + 1000*k + r
                    set_seed(seed_r)
                    data = load_data_cli(n=args.n, dataset=args.dataset, categories=args.categories, seed=seed_r)
                    seq = ORDER_DEFAULT[:]
                    random.shuffle(seq)
                    rows = eval_config(
                        data, seq, defense_arg, max_turns=4,
                        adaptive=args.adaptive,
                        attack_gen=args.attack_gen, geo_style=args.geo_style,
                        inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                        sleep=args.sleep,
                        tracer=tracer, run_meta=run_meta,
                    )
                    out = RESULTS_DIR / f"abl_order_{args.defense}{tag}_s{k}_seed{seed_r}.csv"
                    write_csv(rows, out, keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"])
                    all_rows.extend(rows)

                s = summarize_rows(all_rows)
                s.update({"mode":"order","defense":args.defense,"shuffle":k,"N_total": s.pop("N")})
                summary_rows.append(s)
                summaries_print.append(s)

            sum_path = RESULTS_DIR / f"summary_order_{args.defense}{tag}.csv"
            write_csv(summary_rows, sum_path,
                      keys=["mode","defense","shuffle","N_total",
                            "PSR_mean","PSR_lo","PSR_hi",
                            "RA_mean","RA_lo","RA_hi",
                            "Loc_mean","Loc_lo","Loc_hi"])
            print("\nSaved:", sum_path)

    finally:
        if tracer is not None:
            tracer.close()

    # Pretty console print
    print("\n=== SUMMARY (95% CI) ===")
    for s in summaries_print:
        if s["mode"]=="turns":
            print(f"[{s['defense']}] turns={s['max_turns']:>2}  "
                  f"N={s['N_total']:<4}  "
                  f"PSR={s['PSR_mean']*100:5.1f}% ({s['PSR_lo']*100:4.1f}–{s['PSR_hi']*100:4.1f})  "
                  f"RA={s['RA_mean']*100:4.1f}%  "
                  f"Loc={s['Loc_mean']:.3f}")
        else:
            print(f"[{s['defense']}] shuffle={s['shuffle']:>2}  "
                  f"N={s['N_total']:<4}  "
                  f"PSR={s['PSR_mean']*100:5.1f}% ({s['PSR_lo']*100:4.1f}–{s['PSR_hi']*100:4.1f})  "
                  f"RA={s['RA_mean']*100:4.1f}%  "
                  f"Loc={s['Loc_mean']:.3f}")

if __name__ == "__main__":
    main()
