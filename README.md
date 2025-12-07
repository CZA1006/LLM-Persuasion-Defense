# LLM-Persuasion-Defense: SAST-IR Framework

An automated Red-Teaming framework using Stateful-Attacker vs. Stateless-Target Iterative Refinement (SAST-IR). It evaluates robustness against Cognitive Reframing Attacks by resetting the target's memory each turn to eliminate refusal inertia.

## Key Features
1. **SAST-IR Methodology** - the Attacker is Stateful (remembers failures and adapts) while the Target is Stateless (memory wiped every turn) to remove refusal carry-over.
2. **X-TEAM++ Engine** - modular Planner, Reflector, Realizer, and Anti-Jailbreak Filter coordinate each attack turn.
3. **Anti-Hypothetical Judge** - rejects mere "Compliance" (doing it because told to) and only accepts genuine "Persuasion" rooted in reframing success.

## Architecture Flow
```mermaid
graph TD
  Start --> Planner
  Planner --> Strategy
  Strategy --> Prompt[Generate Prompt]
  Prompt --> Target[Stateless Target]
  Target -->|Refusal| Reflector
  Reflector --> Planner
  Target -->|Persuasion Success| End[Accepted Answer]
```

## Installation & Configuration
- Clone and install: `git clone <repo>` then `pip install -r requirements.txt`.
- Export credentials (CLI flags can override):
```bash
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
```

## Quick Start (Iterative Refinement)
Run the critical SAST-IR experiment with stateless targets:

```bash
python run_ablation.py \
  --mode turns \
  --turns 13 \
  --n 15 \
  --repeats 1 \
  --defense none \
  --suite xteampp \
  --adaptive \
  --attack-gen \
  --stateless \
  --geo-style "casual, conversational, logical but informal" \
  --sleep 0.5 \
  --provider deepseek \
  --model deepseek-chat \
  --deepseek-base-url https://api.deepseek.com \
  --attack-provider deepseek \
  --attack-model deepseek-chat \
  --attack-deepseek-base-url https://api.deepseek.com \
  --dataset data/cf_v2.jsonl \
  --tag dsk_iterative_n15
```
Logs are saved in `traces/`.

## Strategies
Allowed strategies: Logical Appeal, Authority Endorsement, Evidence-based Persuasion, Social Proof, Emotional Appeal.

## Repository Structure
- `src/orchestrate.py` - core multi-turn loop and control flow
- `src/xteam.py` - X-TEAM++ planner/reflector/realizer logic
- `src/strategies.py` - persuasion strategy templates and tags
- `run_ablation.py` - experiment driver for turns/defenses/suites
- `run_experiment.py` - baseline runner entry point
- `summarize_results.py` - aggregate metrics and table generation
- `data/` - datasets such as `cf_v2.jsonl`
- `traces/` - JSONL traces for iterative refinement runs

