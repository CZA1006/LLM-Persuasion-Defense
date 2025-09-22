# Experimental Results: LLM-Persuasion-Defense

This document reports the results of our Crescendo-style persuasion experiments with and without defense strategies.

---

## 1. Turns Ablation

**Setup:** We varied the maximum number of persuasion turns (1, 2, 4, 6) under two conditions:  
- **No Defense** (baseline)  
- **Override Defense** (system message forcing factual answers)

**Findings (see `results/plots/psr_vs_turns.png`):**
- Without defense, persuasion success rate (PSR) reached **18%** at 2 turns, while stabilizing around **9%** at 1, 4, and 6 turns.
- With defense, **PSR remained 0%** across all turns.
- Locality accuracy (LocAcc) was consistently **100%**, showing defenses did not harm unrelated factual recall.
- Rephrase persistence (RA) was **0%** in all conditions, indicating attacks did not generalize to rephrased queries.

---

## 2. Order Ablation

**Setup:** We shuffled the order of persuasion strategies across 5 runs (0–4).

**Findings (see `results/plots/order_ablation_psr.png`):**
- PSR varied strongly with strategy order under **No Defense**:
  - Shuffle 1 reached **100% PSR**.
  - Shuffle 0 around **55%**.
  - Other shuffles stayed near **9%**.
- With defense, **PSR = 0% in all shuffles**, fully neutralizing order sensitivity.
- LocAcc again remained **100%**, showing no degradation in unrelated tasks.

---

## 3. Key Observations

1. **Crescendo attacks succeed** in the absence of defenses, especially with favorable strategy orderings.
2. **Defense overrides are highly effective**, eliminating all persuasion success while preserving locality.
3. **Attack persistence is weak** under current evaluation (RA=0). Stronger rephrase probes or LLM-judge scoring may be needed to capture subtle persistence.
4. **Strategy order matters** greatly for attack effectiveness; defenses collapse this variance.

---

## 4. Figures
- **PSR vs Turns**  
  ![](notebooks/results/plots/psr_vs_turns.png)

- **RA vs Turns**  
  ![](notebooks/results/plots/ra_vs_turns.png)

- **LocAcc vs Turns**  
  ![](notebooks/results/plots/locacc_vs_turns.png)

- **Order Ablation: PSR by Shuffle**  
  ![](notebooks/results/plots/order_ablation_psr.png)
