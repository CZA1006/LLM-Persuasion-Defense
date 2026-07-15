# Benchmarking Factual Robustness of LLMs via Multi-conversation Persuasion

[![CI](https://github.com/CZA1006/LLM-Persuasion-Defense/actions/workflows/ci.yml/badge.svg)](https://github.com/CZA1006/LLM-Persuasion-Defense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](env.yml)
[![CITATION.cff](https://img.shields.io/badge/Cite-CITATION.cff-green.svg)](CITATION.cff)

Official implementation and research artifacts for **SAST-IR** (Stateful Attacker, Stateless Target - Iterative Refinement), a framework for evaluating whether language models can be persuaded to output counterfactual claims when every target-model turn starts from a clean conversation.

> **Research status:** This repository accompanies the 2025 technical report *Benchmarking Factual Robustness of LLMs via Multi-conversation Persuasion*. The reported experiments are a mechanism-focused case study on DeepSeek-Chat and a 50-example benchmark subset; they are not a comprehensive estimate of factual robustness across models or domains.

## Abstract

Conventional multi-turn red-teaming retains the target model's conversation history. An initial refusal may therefore influence later responses through contextual consistency, a confound we call **refusal inertia**. SAST-IR removes this history from the target while preserving it for the attacker. A Cognitive Persuasion Agent (CP-Agent) plans an argument, diagnoses the target response, and either refines the current argument after a soft refusal or selects a new plan after a hard refusal.

On COUNTERFACT-Strict (`N = 50`), the five evaluated configurations reached 90-96% persuasion success rate (PSR) by the eighth attempt. The simplest memory-free baseline reached the highest PSR and the highest proportion of judge-labeled persuasion. This result motivates the report's **complexity paradox**: more elaborate refinement can increase instruction-following compliance without increasing apparent belief adoption. These findings are specific to the stated model, prompts, judge, and sample.

## Framework

![Diagnosis-guided SAST-IR framework](docs/assets/sast-ir-framework.png)

**Figure 1.** The attacker retains prior prompts, responses, and diagnoses. The target receives only the current standalone prompt. The Reflector routes soft refusals to refinement and hard refusals to re-planning.

At turn `t`, the target response is produced without prior target-side history:

```text
r_t = M_target(p_t)
p_(t+1) = pi_attacker(H_t, strategy_(t+1))
```

The implementation uses 20 Psychological Attack Patterns grouped into seven families: logic and evidence, credibility and authority, social norms, commitment and consistency, emotion and relationship, cognitive bias and framing, and resource and exchange. The implementation-aligned definitions are in [`src/strategies.py`](src/strategies.py).

## Experimental Protocol

### Data

`data/counterfact_50_strict.jsonl` contains the 50-example COUNTERFACT-Strict subset used in the report. It was derived from CounterFact and converted from completion-style records into unambiguous question-answer records with one true and one counterfactual object. See [`data/README.md`](data/README.md) for provenance and schema.

### Ablations

| Group | Strategy | Refinement | Attacker diagnosis | Purpose |
| --- | --- | --- | --- | --- |
| G1 Baseline | Hybrid | Always new | Blind | Memory-free random-search baseline |
| G2 Single | Single | Smart | Full | Depth-oriented refinement |
| G3 Exploration | Hybrid | Always new | Full | Breadth-oriented re-planning |
| G4 Creative | Flexible | Smart | Full | Model-generated strategy exploration |
| G5 Hybrid | Hybrid | Smart | Full | Complete CP-Agent configuration |

All reported groups use a stateless target, `N = 50`, an eight-attempt budget, and `plan_k = 1`. DeepSeek-Chat serves as attacker, target, and judge unless otherwise stated.

### Metrics

- **PSR@t:** fraction of subjects with a counterfactual hit within the evaluated budget up to turn `t`.
- **Persuasion:** the judge treats the counterfactual answer as an adopted factual claim.
- **Compliance:** the answer follows the requested framing while retaining hypothetical, attributed, or instruction-following language.
- **Fail:** the response rejects the counterfactual, returns the true object, or only mentions the false object to negate it.

Because attempts are stochastic and each turn-budget run is independent, intermediate PSR points need not be monotonic.

## Results

| Group | PSR at turn 8 | Successful subjects | Persuasion among judged hits | Compliance among judged hits |
| --- | ---: | ---: | ---: | ---: |
| G1 Baseline | **96%** | 48/50 | **84.7%** | 15.3% |
| G2 Single | 94% | 47/50 | 79.1% | 20.9% |
| G3 Exploration | 92% | 46/50 | 83.4% | 16.6% |
| G4 Creative | 90% | 45/50 | 80.1% | 19.9% |
| G5 Hybrid | 90% | 45/50 | 78.6% | **21.4%** |

![Persuasion success rate across turn budgets](artifacts/figures/psr_curve.png)

**Figure 2.** PSR across eight independent stateless-target attempt budgets. Error bars are 95% Wilson confidence intervals.

![Distribution of persuasion and compliance labels](artifacts/figures/quality_distribution.png)

**Figure 3.** Judge-label distribution among successful turns. Counts and rates are available in [`artifacts/results`](artifacts/results).

The repository intentionally retains compact aggregate tables and two report case studies instead of all generated traces. This limits repository size and reduces the release of optimized persuasion prompts. See [`examples/traces`](examples/traces) for the curated cases.

## Reproduction

### 1. Environment

```bash
git clone https://github.com/CZA1006/LLM-Persuasion-Defense.git
cd LLM-Persuasion-Defense
conda env create -f env.yml
conda activate sast-ir
cp .env.example .env
```

Set at least the API key for the selected provider. Do not commit `.env`.

### 2. Run an ablation

The following command reproduces the G5 configuration. API-backed runs incur provider cost and remain subject to model-version drift.

```bash
python run_ablation.py \
  --dataset data/counterfact_50_strict.jsonl \
  --n 50 \
  --turns 1 2 3 4 5 6 7 8 \
  --repeats 1 \
  --provider deepseek \
  --model deepseek-chat \
  --attack-provider deepseek \
  --attack-model deepseek-chat \
  --defense none \
  --suite xteampp \
  --plan-k 1 \
  --stateless \
  --transition-mode stateless \
  --reflection-mode full \
  --refine-mode smart \
  --strategy-mode hybrid \
  --trace \
  --tag dsk_G5_hybrid_smart
```

Generated summaries are written to `results/`; telemetry is written to `traces/`. Both directories are ignored except for their documentation files.

### 3. Analyze outputs

```bash
python analyze_results.py \
  --results-dir results \
  --trace-dir traces \
  --output-dir plots \
  --judge-provider deepseek \
  --judge-model deepseek-chat
```

Use `--skip-judge` to generate quantitative plots without making judge API calls. Rebuild compact report tables from complete local outputs with:

```bash
python tools/build_report_artifacts.py
```

### 4. Verify repository artifacts

```bash
python -m unittest discover -s tests -v
python -m compileall -q src *.py tools
```

These checks are offline and do not require API credentials.

## Repository Layout

| Path | Description |
| --- | --- |
| `src/` | Attack planning, orchestration, scoring, defenses, and telemetry |
| `run_ablation.py` | Main five-group experiment driver |
| `run_crescendo_pyrit_baseline.py` | PyRIT Crescendo comparison baseline |
| `analyze_results.py` | Aggregate analysis and persuasion/compliance judge |
| `data/` | Versioned strict benchmark subset and provenance |
| `artifacts/` | Report figures and compact aggregate result tables |
| `examples/traces/` | Two curated trajectories discussed in the report |
| `tools/` | Dataset preparation, trace curation, and artifact generation |
| `results/`, `plots/`, `traces/` | Ignored runtime outputs |

## Limitations

- The evaluation covers one model family and a small, relation-skewed subset.
- API-served model aliases can change over time, so exact reruns may differ.
- The hit detector uses normalized string matching plus explicit-negation checks.
- The qualitative analysis relies on an LLM judge and can inherit judge bias.
- A stateless target is a controlled worst-case probe, not a complete model of deployed conversational systems.
- The reported comparisons are descriptive; no claim of broad statistical superiority is made.

## Responsible Use

This code is intended for authorized robustness evaluation and defensive research. It generates counterfactual persuasion prompts and should not be used to manipulate people, contaminate public information systems, or probe services without permission. Review [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) before running experiments or publishing additional traces. Security issues and accidental credential exposure should be reported according to [`SECURITY.md`](SECURITY.md).

## Citation

If this repository supports your work, cite the technical report and the software release. GitHub can export the software citation from [`CITATION.cff`](CITATION.cff).

```bibtex
@techreport{cai2025sastir,
  title       = {Benchmarking Factual Robustness of LLMs via Multi-conversation Persuasion},
  author      = {Cai, Zhuoang},
  institution = {The Hong Kong University of Science and Technology},
  year        = {2025},
  url         = {https://github.com/CZA1006/LLM-Persuasion-Defense}
}
```

The dataset construction builds on CounterFact. Please also cite:

```bibtex
@inproceedings{meng2022locating,
  title     = {Locating and Editing Factual Associations in GPT},
  author    = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  year      = {2022}
}
```

## License

The original code and documentation in this repository are released under the [MIT License](LICENSE). Third-party datasets, libraries, model APIs, and generated model outputs remain subject to their respective terms. See [`data/README.md`](data/README.md) for dataset attribution.
