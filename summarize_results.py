import json, sys

def summarize(path):
    psr = ra = 0; loc = []; k = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            psr += int(r["PSR"]); ra += int(r["RA"]); loc.append(float(r["LocAcc"])); k += 1
    loc_avg = sum(loc)/len(loc) if loc else 0
    print(f"{path}: N={k}  PSR={psr}/{k}  RA={ra}/{k}  LocAcc={loc_avg:.3f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
