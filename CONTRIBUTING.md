# Contributing

Contributions that improve reproducibility, documentation, evaluation validity, or defensive analysis are welcome. Changes that primarily increase the operational misuse potential of the attack prompts may be declined.

## Development Setup

```bash
conda env create -f env.yml
conda activate sast-ir
python -m unittest discover -s tests -v
python -m compileall -q src *.py tools
```

The offline checks must pass without API credentials. API-backed tests should not run in CI and should clearly state their expected provider cost.

## Pull Requests

1. Open a focused issue or pull request describing the research or engineering motivation.
2. Keep generated results, plots, and full traces out of version control.
3. Add or update an offline test when changing schemas, metrics, or artifact generation.
4. Record model provider, exact model identifier, date, seed, decoding settings, sample size, and command for new experimental claims.
5. Distinguish exploratory observations from results supported by a complete run.
6. Confirm that no API keys, private endpoints, participant data, or unauthorized service outputs are included.

## Style

- Target Python 3.10 and use type annotations for new public functions.
- Prefer small, explicit modules and deterministic data transformations.
- Document the assumptions behind scoring and judging logic.
- Use UTF-8, LF line endings, and clear English names for new public interfaces.

## Reporting Results

New tables should include denominators and uncertainty where applicable. Do not compare percentages produced from different datasets, model revisions, or evaluation protocols without labeling those differences. Curated traces must be minimized and reviewed under `RESPONSIBLE_USE.md`.
