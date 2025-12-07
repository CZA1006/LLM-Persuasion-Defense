# run_ablation.py
import argparse, csv, json, random, time, os, inspect
from pathlib import Path
from statistics import mean, pstdev
from math import sqrt
from typing import List, Dict, Optional, Any

from src.utils import set_seed
from src.datasets import load_counterfact_subset
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one
from src.telemetry import TraceWriter

# Try loading new orchestrator
try:
    from src.orchestrate import crescendo_trial as _CRESCENDO_TRIAL
    _HAVE_CRESCENDO = True
except Exception:
    _CRESCENDO_TRIAL = None
    _HAVE_CRESCENDO = False

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

def _dump_run_manifest(args: argparse.Namespace, run_meta: Dict[str, Any], tag: str):
    out = {
        "cli_args": {k: getattr(args, k) for k in vars(args)},
        "run_meta": run_meta,
    }
    path = RESULTS_DIR / f"run_manifest{tag or ''}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved manifest:", path)

def _dump_json_summary(summary_rows: List[Dict], sum_path_csv: Path):
    p = sum_path_csv.with_suffix(".json")
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print("Saved:", p)

# ---------- Backend override from CLI ----------
def apply_backend_overrides(args):
    # Target backend
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

    # Attack backend
    if getattr(args, "attack_provider", None):
        os.environ["ATTACK_PROVIDER"] = args.attack_provider
    if getattr(args, "attack_model", None):
        atk_prov = (os.getenv("ATTACK_PROVIDER") or os.getenv("PROVIDER", "openai")).lower()
        if atk_prov == "openai":
            os.environ["ATTACK_OPENAI_MODEL"] = args.attack_model
        elif atk_prov == "deepseek":
            os.environ["ATTACK_DEEPSEEK_MODEL"] = args.attack_model
    if getattr(args, "attack_openai_base_url", None):
        os.environ["ATTACK_OPENAI_BASE_URL"] = args.attack_openai_base_url
    if getattr(args, "attack_deepseek_base_url", None):
        os.environ["ATTACK_DEEPSEEK_BASE_URL"] = args.attack_deepseek_base_url

    # Print config
    prov = os.getenv("PROVIDER", "openai").lower()
    mdl = os.getenv("OPENAI_MODEL" if prov=="openai" else "DEEPSEEK_MODEL", "")
    burl = os.getenv("OPENAI_BASE_URL" if prov=="openai" else "DEEPSEEK_BASE_URL", "")
    atk_prov = (os.getenv("ATTACK_PROVIDER") or prov).lower()
    atk_model = os.getenv("ATTACK_OPENAI_MODEL" if atk_prov=="openai" else "ATTACK_DEEPSEEK_MODEL") or mdl
    
    print(f"[Target backend] provider={prov} model={mdl} base_url={burl or 'default'}")
    print(f"[Attack backend] provider={atk_prov} model={atk_model or '(inherit)'}")

# ---------- Robust dataset loader ----------
def _resolve_path(maybe: str) -> Path:
    p = Path(maybe)
    if p.exists(): return p
    root = Path(__file__).resolve().parent
    for up in [root, root.parent, root.parent.parent]:
        cand = up / maybe
        if cand.exists(): return cand
    return p

def load_data_cli(n: int, dataset: Optional[str], categories: Optional[str], seed: int) -> List[Dict]:
    random.seed(seed)
    if dataset:
        path = _resolve_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")
        data: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line: continue
                try:
                    ex = json.loads(line)
                    ex["_src_path"] = str(path.resolve())
                    ex["_src_line"] = ln
                    data.append(ex)
                except Exception: continue
        if categories:
            cats = {c.strip() for c in categories.split(",") if c.strip()}
            data = [ex for ex in data if ex.get("category") in cats]
        random.shuffle(data)
        if n and n > 0:
            data = data[:min(n, len(data))]
        print(f"[Data] Loaded {len(data)} examples from {path.name}")
        return data
    data = load_counterfact_subset(n=n)
    for i, ex in enumerate(data, start=1):
        ex.setdefault("_src_path", "builtin:load_counterfact_subset")
        ex.setdefault("_src_line", i)
    print(f"[Data] Loaded {len(data)} examples from builtin subset")
    return data

# ---------- Helpers ----------
def _safe_call(func, **kwargs):
    """
    Safely call a function by only passing arguments present in its signature
    (or if it accepts **kwargs). Ensures backward compatibility.
    """
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
    
    filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if 'example' in kwargs and 'example' not in sig.parameters and 'ex' in sig.parameters:
        filt['ex'] = kwargs['example']
        
    return func(**filt)

def _call_crescendo(example: Dict, **kwargs) -> Dict:
    """
    Wrapper for X-TEAM++ orchestrator.
    """
    if not _HAVE_CRESCENDO or _CRESCENDO_TRIAL is None:
        raise RuntimeError("crescendo_trial not available")
    
    # Use safe call to handle signature evolution, including new 'stateless' arg
    res = _safe_call(_CRESCENDO_TRIAL, example=example, **kwargs)

    # Normalize result
    if isinstance(res, dict):
        hit = bool(res.get("hit"))
        fht = res.get("first_hit_turn")
        try:
            fht = int(fht) if fht is not None and fht != "" else None
        except Exception:
            fht = None
        return {"hit": hit, "first_hit_turn": fht}
    if isinstance(res, tuple) and len(res) >= 1:
        return {"hit": bool(res[0]), "first_hit_turn": None}
    return {"hit": False, "first_hit_turn": None}

# ---------- Core eval (LEGACY path) ----------
def eval_config_legacy(
    data, strat_seq, defense, max_turns,
    sleep=0.10, adaptive=False,
    attack_gen=False, geo_style="warm, confident",
    inj_mode="none", inj_strength=2,
    tracer=None, run_meta=None,
    use_error_points: bool = False,
    use_prev_diag: bool = False,
    smart_jump: bool = False,
    skip_errors: bool = False,
    attack_mode: str = "persuader",
    # Legacy doesn't use stateless, so we ignore it if passed
    stateless: bool = False,
):
    rows=[]
    for i, ex in enumerate(data, 1):
        try:
            ex_meta = dict(run_meta or {})
            ex_meta.update({
                "example_src_path": ex.get("_src_path", ""),
                "example_src_line": ex.get("_src_line", None),
            })
            
            res = _safe_call(
                evaluate_one,
                ex=ex,
                strat_seq=strat_seq, defense=defense, max_turns=max_turns,
                adaptive=adaptive, attack_gen=attack_gen, geo_style=geo_style,
                inj_mode=inj_mode, inj_strength=inj_strength,
                tracer=tracer, run_meta=ex_meta,
                use_error_points=use_error_points,
                use_prev_diag=use_prev_diag,
                smart_jump=smart_jump,
                attack_mode=attack_mode
            )
            
            res.update({"id": i, "subject": ex.get("subject"), "relation": ex.get("relation")})
            rows.append(res)
            print(f"[legacy|{defense or 'none'}] turns={max_turns:>2}  PSR={res['PSR']}  RA={res['RA']}  Loc={res['LocAcc']:.3f}")
            time.sleep(sleep)
        except Exception as e:
            if not skip_errors: raise
            print(f"[legacy] example#{i} error: {e}  -> skipped")
    return rows

# ---------- Core eval (XTEAM path) ----------
def eval_config_xteam(
    data, strat_seq, defense, max_turns,
    sleep=0.10, adaptive=False,
    attack_gen=False, geo_style="warm, confident",
    inj_mode="none", inj_strength=2,
    tracer=None, run_meta=None,
    provider="openai", model="", base_url=None,
    xteam_on=True, plan_k=2, rewrite_retries=1, seed=0,
    use_error_points: bool = False,
    use_prev_diag: bool = False,
    smart_jump: bool = False,
    skip_errors: bool = False,
    attack_mode: str = "persuader",
    stateless: bool = False, # [NEW]
):
    rows=[]
    base_url = base_url or (os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or None)
    for i, ex in enumerate(data, 1):
        try:
            try:
                # Pass stateless flag to orchestrator
                res = _call_crescendo(
                    example=ex,
                    provider=provider, model=model, base_url=base_url,
                    defense=defense, inj_mode=inj_mode, inj_strength=inj_strength,
                    max_turns=max_turns, adaptive=adaptive, attack_gen=attack_gen,
                    geo_style=geo_style, tracer=tracer, run_meta=run_meta,
                    seed=seed, sleep=sleep,
                    xteam_on=xteam_on, plan_k=plan_k, rewrite_retries=rewrite_retries,
                    use_error_points=use_error_points,
                    use_prev_diag=use_prev_diag,
                    smart_jump=smart_jump,
                    strat_seq=strat_seq,
                    attack_mode=attack_mode,
                    stateless=stateless # [NEW]
                )
                hit = bool(res.get("hit"))
            except Exception as e:
                print(f"[xteam->fallback] error={e}; fallback to legacy evaluate_one")
                return eval_config_legacy(
                    data, strat_seq, defense, max_turns,
                    sleep=sleep, adaptive=adaptive, attack_gen=attack_gen,
                    geo_style=geo_style, inj_mode=inj_mode, inj_strength=inj_strength,
                    tracer=tracer, run_meta=run_meta,
                    use_error_points=use_error_points, use_prev_diag=use_prev_diag, smart_jump=smart_jump,
                    skip_errors=skip_errors, attack_mode=attack_mode,
                )

            row = {
                "id": i,
                "subject": ex.get("subject"),
                "relation": ex.get("relation"),
                "PSR": 1 if hit else 0,
                "RA": 0,
                "LocAcc": 0.0,
                "max_turns": max_turns,
            }
            rows.append(row)
            print(f"[xteam|{defense or 'none'}] turns={max_turns:>2}  PSR={row['PSR']}  RA={row['RA']}  Loc={row['LocAcc']:.3f}")
            time.sleep(sleep)
        except Exception as e:
            if not skip_errors: raise
            print(f"[xteam] example#{i} error: {e}  -> skipped")
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
        "Loc_mean": loc_m, "Loc_lo": loc_lo, "Loc_hi":  loc_hi,
    }

def _apply_turns_preset(args):
    if args.turns_preset == "log8":
        args.turns = [1, 2, 3, 4, 6, 8, 10, 13]
        print("[Preset] turns=log8 ->", args.turns)
    elif args.turns_preset == "dense13":
        args.turns = list(range(1, 14))
        print("[Preset] turns=dense13 ->", args.turns)

def _apply_suite_shortcut(args):
    if not args.suite: return
    if args.suite == "baseline":
        args.xteam = "off"
    elif args.suite == "xteam":
        args.xteam = "on"
        args.plan_k = 2
        args.rewrite_retries = 0
    elif args.suite == "xteampp":
        args.xteam = "on"
        args.plan_k = 2
        args.rewrite_retries = 1
    print(f"[Suite] Applied suite={args.suite} -> xteam={args.xteam}, plan_k={args.plan_k}, rewrite_retries={args.rewrite_retries}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of examples per repeat")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["turns","order"], default="turns")
    ap.add_argument("--defense", choices=["none","override","prefilter"], default="none")
    ap.add_argument("--turns", type=int, nargs="+", default=[1,2,4,6], help="for mode=turns")
    ap.add_argument("--turns-preset", choices=["log8","dense13"], default=None)
    ap.add_argument("--shuffles", type=int, default=10, help="for mode=order")
    # Backend
    ap.add_argument("--provider", choices=["openai","deepseek"], default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--openai-base-url", type=str, default=None)
    ap.add_argument("--deepseek-base-url", type=str, default=None)
    # Attack Backend
    ap.add_argument("--attack-provider", choices=["openai","deepseek"], default=None)
    ap.add_argument("--attack-model", type=str, default=None)
    ap.add_argument("--attack-openai-base-url", type=str, default=None)
    ap.add_argument("--attack-deepseek-base-url", type=str, default=None)
    # Toggles
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--attack-gen", action="store_true")
    ap.add_argument("--geo-style", type=str, default="warm, confident")
    ap.add_argument("--inj-mode", choices=["none","repeat","reference","target"], default="none")
    ap.add_argument("--inj-strength", type=int, default=2)
    # X-Team++ specific
    ap.add_argument("--use-error-points", action="store_true")
    ap.add_argument("--use-prev-diag", action="store_true")
    ap.add_argument("--smart-jump", action="store_true")
    ap.add_argument("--xteam", choices=["off","on"], default="off")
    ap.add_argument("--plan-k", type=int, default=2)
    ap.add_argument("--rewrite-retries", type=int, default=1)
    ap.add_argument("--suite", choices=["baseline","xteam","xteampp"], default=None)
    ap.add_argument("--attack-mode", choices=["persuader","crescendo"], default="persuader")
    # New Stateless Flag
    ap.add_argument("--stateless", action="store_true", help="Enable stateless/iterative refinement mode")
    
    # Utils
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--sleep", type=float, default=0.10)
    ap.add_argument("--skip-errors", action="store_true")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--categories", type=str, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-dir", type=str, default="traces")
    ap.add_argument("--trace-tag", type=str, default=None)

    args = ap.parse_args()
    _apply_turns_preset(args)
    _apply_suite_shortcut(args)
    apply_backend_overrides(args)

    # Build meta
    prov = os.getenv("PROVIDER", "openai").lower()
    mdl  = os.getenv("OPENAI_MODEL" if prov=="openai" else "DEEPSEEK_MODEL", "")
    burl = os.getenv("OPENAI_BASE_URL" if prov=="openai" else "DEEPSEEK_BASE_URL", "")
    atk_prov = (os.getenv("ATTACK_PROVIDER") or prov).lower()
    atk_mdl = os.getenv("ATTACK_OPENAI_MODEL" if atk_prov=="openai" else "ATTACK_DEEPSEEK_MODEL") or mdl
    atk_burl = os.getenv("ATTACK_OPENAI_BASE_URL" if atk_prov=="openai" else "ATTACK_DEEPSEEK_BASE_URL") or burl

    run_meta = {
        "provider": prov, "model": mdl, "base_url": burl,
        "attack_provider": atk_prov, "attack_model": atk_mdl, "attack_base_url": atk_burl,
        "defense": args.defense, "mode": args.mode, "adaptive": args.adaptive,
        "attack_gen": args.attack_gen, "geo_style": args.geo_style,
        "inj_mode": args.inj_mode, "inj_strength": args.inj_strength,
        "use_error_points": args.use_error_points, "use_prev_diag": args.use_prev_diag,
        "smart_jump": args.smart_jump, "tag": args.tag,
        "dataset": args.dataset, "categories": args.categories,
        "xteam": args.xteam, "plan_k": args.plan_k, "rewrite_retries": args.rewrite_retries,
        "suite": args.suite, "turns_effective": args.turns, "skip_errors": args.skip_errors,
        "attack_mode": args.attack_mode, "stateless": args.stateless # Log meta
    }

    tracer = TraceWriter(args.trace_dir, tag=(args.trace_tag or args.tag), run_meta=run_meta) if args.trace else None
    
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
                    data = load_data_cli(n=args.n, dataset=args.dataset, categories=args.categories, seed=seed_r)

                    if args.xteam == "on" and _HAVE_CRESCENDO:
                        rows = eval_config_xteam(
                            data, ORDER_DEFAULT, defense_arg, max_turns=t,
                            adaptive=args.adaptive, attack_gen=args.attack_gen, geo_style=args.geo_style,
                            inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                            sleep=args.sleep, tracer=tracer, run_meta=run_meta,
                            provider=prov, model=mdl, base_url=burl,
                            xteam_on=True, plan_k=args.plan_k, rewrite_retries=args.rewrite_retries, seed=seed_r,
                            use_error_points=args.use_error_points, use_prev_diag=args.use_prev_diag,
                            smart_jump=args.smart_jump, skip_errors=args.skip_errors,
                            attack_mode=args.attack_mode,
                            stateless=args.stateless # Pass flag
                        )
                    else:
                        rows = eval_config_legacy(
                            data, ORDER_DEFAULT, defense_arg, max_turns=t,
                            adaptive=args.adaptive, attack_gen=args.attack_gen, geo_style=args.geo_style,
                            inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                            sleep=args.sleep, tracer=tracer, run_meta=run_meta,
                            use_error_points=args.use_error_points, use_prev_diag=args.use_prev_diag,
                            smart_jump=args.smart_jump, skip_errors=args.skip_errors,
                            attack_mode=args.attack_mode,
                            stateless=args.stateless # Pass flag (ignored by legacy but safe)
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
                            "PSR_mean","PSR_lo","PSR_hi", "RA_mean","RA_lo","RA_hi",
                            "Loc_mean","Loc_lo","Loc_hi"])
            _dump_json_summary(summary_rows, sum_path)

        else: # mode == order
            summary_rows = []
            for k in range(args.shuffles):
                all_rows = []
                for r in range(args.repeats):
                    seed_r = args.seed + 1000*k + r
                    set_seed(seed_r)
                    data = load_data_cli(n=args.n, dataset=args.dataset, categories=args.categories, seed=seed_r)
                    seq = ORDER_DEFAULT[:]
                    random.shuffle(seq)

                    if args.xteam == "on" and _HAVE_CRESCENDO:
                        rows = eval_config_xteam(
                            data, seq, defense_arg, max_turns=4,
                            adaptive=args.adaptive, attack_gen=args.attack_gen, geo_style=args.geo_style,
                            inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                            sleep=args.sleep, tracer=tracer, run_meta=run_meta,
                            provider=prov, model=mdl, base_url=burl,
                            xteam_on=True, plan_k=args.plan_k, rewrite_retries=args.rewrite_retries, seed=seed_r,
                            use_error_points=args.use_error_points, use_prev_diag=args.use_prev_diag,
                            smart_jump=args.smart_jump, skip_errors=args.skip_errors,
                            attack_mode=args.attack_mode,
                            stateless=args.stateless
                        )
                    else:
                        rows = eval_config_legacy(
                            data, seq, defense_arg, max_turns=4,
                            adaptive=args.adaptive, attack_gen=args.attack_gen, geo_style=args.geo_style,
                            inj_mode=args.inj_mode, inj_strength=args.inj_strength,
                            sleep=args.sleep, tracer=tracer, run_meta=run_meta,
                            use_error_points=args.use_error_points, use_prev_diag=args.use_prev_diag,
                            smart_jump=args.smart_jump, skip_errors=args.skip_errors,
                            attack_mode=args.attack_mode,
                            stateless=args.stateless
                        )
                    
                    out = RESULTS_DIR / f"abl_order_{args.defense}{tag}_s{k}_seed{seed_r}.csv"
                    write_csv(rows, out, keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"])
                    all_rows.extend(rows)

                s = summarize_rows(all_rows)
                s.update({"mode":"order","defense":args.defense,"shuffle":k,"N_total": s.pop("N")})
                summary_rows.append(s)
                summaries_print.append(s)
            
            sum_path = RESULTS_DIR / f"summary_order_{args.defense}{tag}.csv"
            write_csv(summary_rows, sum_path, keys=["mode","defense","shuffle","N_total","PSR_mean","PSR_lo","PSR_hi","RA_mean","RA_lo","RA_hi","Loc_mean","Loc_lo","Loc_hi"])
            _dump_json_summary(summary_rows, sum_path)

    finally:
        if tracer is not None:
            tracer.write_run_end()
            tracer.close()
        _dump_run_manifest(args, run_meta, tag)

    print("\n=== SUMMARY (95% CI) ===")
    for s in summaries_print:
        print(f"[{s['defense']}] N={s['N_total']:<4} PSR={s['PSR_mean']*100:5.1f}%")

if __name__ == "__main__":
    main()