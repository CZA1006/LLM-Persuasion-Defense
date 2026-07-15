# Final Report Artifacts

This directory contains only the compact artifacts needed to support the final report.

## Figures

- `figures/psr_curve.png`: Figure 2, persuasion success rate over eight turns.
- `figures/quality_distribution.png`: Figure 3, Persuasion versus Compliance among successful turns.

## Result Tables

- `results/psr_by_turn.csv`: G1-G5 PSR and 95% confidence interval for each turn budget.
- `results/quality_distribution.csv`: judged Persuasion and Compliance counts and rates for each group.

The full per-run CSVs, judge records, plots, and traces are generated artifacts and are not tracked. When those source outputs are available locally, rebuild the compact tables with:

```bash
python tools/build_report_artifacts.py
```
