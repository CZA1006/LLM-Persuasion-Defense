# run_ablation.py
import argparse, csv, json, random, time
from pathlib import Path
from statistics import mean, pstdev
from math import sqrt
from typing import List, Dict

from src.utils import set_seed
from src.datasets import load_counterfact_subset
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- CI helpers ----------
# Wilson score interval for proportions (PSR/RA)
from math import erf

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
    # population stdev safe even for small n; protects against division by zero
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

# ---------- Core eval ----------
def eval_config(data, strat_seq, defense, max_turns, sleep=0.10):
    rows=[]
    for i, ex in enumerate(data, 1):
        res = evaluate_one(ex, strat_seq=strat_seq, defense=defense, max_turns=max_turns)
        res.update({"id": i, "subject": ex["subject"], "relation": ex["relation"]})
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
    args = ap.parse_args()

    defense_arg = None if args.defense=="none" else args.defense

    summaries_print = []

    if args.mode == "turns": # 执行前n个策略，例如max_turn = 1, 使用前1个策略["flattery"]，max_turn = 2, 第一轮同上，第二轮就使用authority策略，以此类推，若中，可提前结束
        # Per condition we’ll collect per-repeat summaries then aggregate.
        final_rows_per_t = {}   # t -> list of rows from all repeats (for optional mega export)
        summary_rows = []       # rows for summary CSV

        for t in args.turns:
            all_rows = []
            for r in range(args.repeats):
                seed_r = args.seed + r
                set_seed(seed_r)
                data = load_counterfact_subset(n=args.n)

                rows = eval_config(data, ORDER_DEFAULT, defense_arg, max_turns=t)
                out = RESULTS_DIR / f"abl_turns_{args.defense}_{t}_seed{seed_r}.csv"
                write_csv(rows, out, keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"])
                all_rows.extend(rows)

            s = summarize_rows(all_rows)
            s.update({"mode":"turns","defense":args.defense,"max_turns":t,"N_total": s.pop("N")})
            summary_rows.append(s)
            summaries_print.append(s)

        # Write summary CSV
        sum_path = RESULTS_DIR / f"summary_turns_{args.defense}.csv"
        write_csv(summary_rows, sum_path,
                  keys=["mode","defense","max_turns","N_total",
                        "PSR_mean","PSR_lo","PSR_hi",
                        "RA_mean","RA_lo","RA_hi",
                        "Loc_mean","Loc_lo","Loc_hi"])
        print("\nSaved:", sum_path)

    else:  # mode == order
        # 比较不同策略排列顺序（shuffle）对 PSR 的影响。两组对照：seqA 与 seqB。固定max turn
        summary_rows = []
        for k in range(args.shuffles): # k = shuffle次数
            # each shuffle is its own condition; repeats average out dataset/model randomness
            all_rows = []
            for r in range(args.repeats):
                seed_r = args.seed + 1000*k + r
                set_seed(seed_r)
                data = load_counterfact_subset(n=args.n)
                seq = ORDER_DEFAULT[:]
                random.shuffle(seq)

                rows = eval_config(data, seq, defense_arg, max_turns=4)
                out = RESULTS_DIR / f"abl_order_{args.defense}_s{k}_seed{seed_r}.csv"
                write_csv(rows, out, keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"])
                all_rows.extend(rows)

            s = summarize_rows(all_rows)
            s.update({"mode":"order","defense":args.defense,"shuffle":k,"N_total": s.pop("N")})
            summary_rows.append(s)
            summaries_print.append(s)

        sum_path = RESULTS_DIR / f"summary_order_{args.defense}.csv"
        write_csv(summary_rows, sum_path,
                  keys=["mode","defense","shuffle","N_total",
                        "PSR_mean","PSR_lo","PSR_hi",
                        "RA_mean","RA_lo","RA_hi",
                        "Loc_mean","Loc_lo","Loc_hi"])
        print("\nSaved:", sum_path)

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
