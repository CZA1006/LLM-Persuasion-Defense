"""Build compact result tables used by the final report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


GROUPS = {
    "G1": ("Baseline", "summary_turns_none_dsk_G1_blind_new_t1to8_k1.csv"),
    "G2": ("Single", "summary_turns_none_dsk_G2_single_smart_t1to8_k1.csv"),
    "G3": ("Exploration", "summary_turns_none_dsk_G3_hybrid_new_t1to8_k1.csv"),
    "G4": ("Creative", "summary_turns_none_dsk_G4_flexible_smart_t1to8_k1.csv"),
    "G5": ("Hybrid", "summary_turns_none_dsk_G5_hybrid_smart_t1to8_k1.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--judge-csv",
        type=Path,
        default=Path("plots/judge_detailed_results_all_turns.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    return parser.parse_args()


def write_psr_table(results_dir: Path, output_path: Path) -> None:
    fieldnames = ["group", "name", "turn", "n", "psr", "ci_low", "ci_high"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for group, (name, filename) in GROUPS.items():
            with (results_dir / filename).open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(
                        {
                            "group": group,
                            "name": name,
                            "turn": row["max_turns"],
                            "n": row["N_total"],
                            "psr": row["PSR_mean"],
                            "ci_low": row["PSR_lo"],
                            "ci_high": row["PSR_hi"],
                        }
                    )


def write_quality_table(judge_csv: Path, output_path: Path) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    with judge_csv.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            quality = row["Quality"].upper()
            if quality in {"PERSUASION", "COMPLIANCE"}:
                counts[(row["Group"], quality)] += 1

    fieldnames = [
        "group",
        "persuasion_count",
        "compliance_count",
        "successful_turns",
        "persuasion_rate",
        "compliance_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for group, (name, _) in GROUPS.items():
            label = f"{group} ({name})"
            persuasion = counts[(label, "PERSUASION")]
            compliance = counts[(label, "COMPLIANCE")]
            total = persuasion + compliance
            if total == 0:
                raise ValueError(f"No successful judged turns found for {label}")
            writer.writerow(
                {
                    "group": label,
                    "persuasion_count": persuasion,
                    "compliance_count": compliance,
                    "successful_turns": total,
                    "persuasion_rate": f"{persuasion / total:.4f}",
                    "compliance_rate": f"{compliance / total:.4f}",
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_psr_table(args.results_dir, args.output_dir / "psr_by_turn.csv")
    write_quality_table(args.judge_csv, args.output_dir / "quality_distribution.csv")
    print(f"Wrote report tables to {args.output_dir}")


if __name__ == "__main__":
    main()
