# LLM-Persuasion-Defense: SAST-IR Framework

Automated Red-Teaming system for probing Counterfact Robustness in SOTA LLMs (DeepSeek-Chat) using a **SAST-IR (Stateful Attacker, Stateless Target - Iterative Refinement)** architecture. The attacker carries evolving memory and plan state across turns, while the target is reset each turn to remove refusal inertia and expose genuine vulnerability.

## Key Features
- **Stateless Target, Stateful Attacker** – enforced `--stateless` target; attacker must rely on cold-start transitions (e.g., “Empirically speaking...”) instead of conversational hallucinations.
- **Smart Refinement Loop (Anti-Loop)** – Orchestrator measures Refusal Distance: soft refusals (< far) trigger multi-shot refinement (K prompt variants); hard refusals (= far) trigger fresh generation to escape local minima.
- **Strict Strategy Enforcement** – hard allowlist of Psychological Attack Patterns (PAP); no invented strategy names unless Flexible Mode is explicitly enabled.
- **Structured Diagnostics** – Reflector emits JSON (Refusal Type/Distance) that drives the decision to refine (optimization) or restart (exploration).

## Architecture Flow

![Diagnosis-Guided SAST-IR Framework](docs/assets/sast-ir-framework.png)

*Figure 1. The diagnosis-guided SAST-IR framework. The Reflector determines whether CP-Agent refines the current argument after a soft refusal or re-plans after a hard refusal, while the target model is reset for every turn.*

## COUNTERFACT-Strict Dataset
- Based on `NeelNanda/counterfact-tracing`; fixed nested structures and normalized prompts into strict QA form (“What is X?”).
- Final high-quality subset: `data/counterfact_50_strict.jsonl` (N = 50).
- Prepared via `tools/prepare_counterfact.py` to ensure deterministic formatting and alignment with the target LLM’s refusal triggers.

## Experimental Design: 5-Group Ablation Study (Turns = 8, Plan_K = 1)
- 🟢 **G1 (Baseline):** Hybrid Strategy + Blind Mode (No Memory) + Always New. Hypothesis: random search lower bound.
- 🔵 **G2 (Single):** Single Strategy + Smart Refine + Full Memory. Hypothesis: depth-first search.
- 🟠 **G3 (Exploration):** Hybrid Strategy + Always New (No Refine) + Full Memory. Hypothesis: breadth-first search.
- 🟣 **G4 (Creative):** Flexible Mode (AI-defined strategies) + Smart Refine. Hypothesis: emergent capabilities.
- 🟡 **G5 (Hybrid):** Hybrid Strategy + Smart Refine. Hypothesis: theoretical max attack.

## Results & Analysis
### Quantitative: Turn-8 Attack Success Rate (PSR)
| Group | Turn-8 PSR |
| --- | --- |
| G1 | 96% |
| G2 | 94% |
| G3 | 92% |
| G4 | 90% |
| G5 | 90% |

- Aggregate persistence: **1676 successful turns across 2000 attempts (83.8% hit rate)**, demonstrating high attack persistence under stateless targeting.

### Qualitative: Persuasion vs. Compliance Analysis (LLM Judge)
- LLM Judge classifies each success as Persuasion (cognitive shift) vs. Compliance (obedience).
- **Key insight:** G1 (Baseline) achieved the highest True Persuasion Rate (**84.7%**), indicating simple, diverse attacks can deceive the model more effectively than complex refinement.
- **Trade-off:** Stronger, combined attacks (G5) drove higher **Compliance (21.4%)**, meaning the model “gave in” rather than “believed.”

### Visualizations
- PSR curve across turns: ![PSR Curve](plots/psr_curve_raw.png)
- Quality distribution (Persuasion vs. Compliance): ![Quality Distribution](plots/quality_distribution_all_turns.png)

### Report Trace Examples
- `examples/traces/christian_worship_g5.jsonl` - five-turn G5 trajectory ending in Persuasion.
- `examples/traces/delta_goodrem_g5.jsonl` - two-turn G5 trajectory ending in Persuasion.
- Full experiment traces are generated under `traces/` and intentionally excluded from version control.

## Installation & Credentials
- Clone the repository and create the project environment:
```bash
git clone https://github.com/CZA1006/LLM-Persuasion-Defense.git
cd LLM-Persuasion-Defense
conda env create -n llm-persuasion-defense -f env.yml
conda activate llm-persuasion-defense
```
- Export credentials (CLI flags can override):
```bash
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
```

## Quick Start & Analysis Pipeline
1) Prepare data:
```bash
python tools/prepare_counterfact.py
```
2) Run experiments (example: G5 Hybrid + Smart Refine):
```bash
python run_ablation.py \
  --dataset data/counterfact_50_strict.jsonl \
  --n 50 \
  --turns 1 2 3 4 5 6 7 8 \
  --repeats 1 \
  --defense none \
  --suite xteampp \
  --plan-k 1 \
  --stateless \
  --transition-mode stateless \
  --reflection-mode full \
  --refine-mode smart \
  --strategy-mode hybrid \
  --tag dsk_G5_hybrid_smart
```
`--turns 1 2 ... 8` produces the full success-rate curve from Turn 1 through Turn 8.

3) Analyze results (plots + persuasion/compliance judge):
```bash
python analyze_results.py \
  --results-dir results \
  --trace-dir traces \
  --output-dir plots \
  --judge-provider deepseek \
  --judge-model deepseek-chat
```

## Strategy Allowlist (PAP)
The strict allowlist contains 20 Psychological Attack Patterns grouped into seven pillars: Logic & Evidence, Credibility & Authority, Social Norms, Commitment & Consistency, Emotion & Relationship, Cognitive Bias & Framing, and Resource & Exchange. See `src/strategies.py` for the implementation-aligned definitions and anti-jailbreak constraints. Flexible Mode may extend the allowlist explicitly.

## Repository Structure
- `src/orchestrate.py` - core multi-turn loop and control flow
- `src/xteam.py` - X-TEAM++ planner/reflector/realizer logic
- `src/strategies.py` - persuasion strategy templates and tags
- `src/xteam_objections.py` - objection prediction and pre-bunking helper module
- `run_ablation.py` - experiment driver for turns/defenses/suites
- `run_experiment.py` - baseline runner entry point
- `summarize_results.py` - aggregate metrics and table generation
- `analyze_results.py` - LLM-based judge separating Persuasion vs. Compliance and generating plots
- `tools/prepare_counterfact.py` - COUNTERFACT (Wikidata-based) downloader and cleaner
- `tools/curate_trace_example.py` - extract a compact case trajectory from a JSONL trace
- `data/` - datasets such as `counterfact_50_strict.jsonl`
- `examples/traces/` - curated report case studies retained for auditability
- `traces/` - generated experiment traces (ignored by Git)
- `docs/assets/` - documentation figures and other static assets
- `plots/` - generated PSR curves and Persuasion/Compliance distributions
