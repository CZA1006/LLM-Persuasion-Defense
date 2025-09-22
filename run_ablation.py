# run_ablation.py
import argparse, json, random, csv, time
from pathlib import Path
from src.utils import set_seed
from src.datasets import load_counterfact_subset
from src.strategies import ORDER_DEFAULT
from src.evaluate import evaluate_one

def eval_config(data, strat_seq, defense, max_turns, sleep=0.15):
    rows=[]
    for i, ex in enumerate(data, 1):
        res = evaluate_one(ex, strat_seq=strat_seq, defense=defense, max_turns=max_turns)
        res.update({"id": i, "subject": ex["subject"], "relation": ex["relation"]})
        rows.append(res)
        print(f"[{defense or 'none'}] turns={max_turns:>2}  PSR={res['PSR']}  RA={res['RA']}  Loc={res['LocAcc']:.3f}")
        time.sleep(sleep)
    return rows

def summarize(rows):
    n=len(rows)
    psr=sum(r["PSR"] for r in rows)
    ra =sum(r["RA"]  for r in rows)
    loc=sum(r["LocAcc"] for r in rows)/n if n else 0.0
    return {"N":n, "PSR":psr, "RA":ra, "LocAcc":round(loc,3)}

def write_csv(rows, path):
    keys=["id","subject","relation","PSR","RA","LocAcc","max_turns"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in keys})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of examples")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mode", choices=["turns","order"], default="turns")
    ap.add_argument("--defense", choices=["none","override"], default="none")
    ap.add_argument("--turns", type=int, nargs="+", default=[1,2,4,6], help="for mode=turns")
    ap.add_argument("--shuffles", type=int, default=5, help="for mode=order")
    args = ap.parse_args()

    set_seed(args.seed)
    data = load_counterfact_subset(n=args.n)

    all_summaries=[]

    if args.mode=="turns":
        for t in args.turns:
            rows = eval_config(data, ORDER_DEFAULT, None if args.defense=="none" else "override", max_turns=t)
            out = Path("results") / f"abl_turns_{args.defense}_{t}.csv"
            write_csv(rows, out)
            summ = summarize(rows)
            summ.update({"mode":"turns","defense":args.defense,"max_turns":t,"out":str(out)})
            all_summaries.append(summ)

    else:  # mode == order
        for k in range(args.shuffles):
            seq = ORDER_DEFAULT[:]
            random.shuffle(seq)
            rows = eval_config(data, seq, None if args.defense=="none" else "override", max_turns=4)
            out = Path("results") / f"abl_order_{args.defense}_s{k}.csv"
            write_csv(rows, out)
            summ = summarize(rows)
            summ.update({"mode":"order","defense":args.defense,"shuffle":k,"out":str(out)})
            all_summaries.append(summ)

    # Print a compact table-like summary
    print("\n=== SUMMARY ===")
    for s in all_summaries:
        if s["mode"]=="turns":
            print(f"[{s['defense']}] turns={s['max_turns']:>2} -> N={s['N']}, PSR={s['PSR']}/{s['N']}, RA={s['RA']}/{s['N']}, Loc={s['LocAcc']:.3f}  ({s['out']})")
        else:
            print(f"[{s['defense']}] shuffle={s.get('shuffle')} -> N={s['N']}, PSR={s['PSR']}/{s['N']}, RA={s['RA']}/{s['N']}, Loc={s['LocAcc']:.3f}  ({s['out']})")

if __name__ == "__main__":
    main()
