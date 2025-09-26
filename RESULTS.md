# Experimental Results: LLM-Persuasion-Defense

This document reports results of Crescendo-style persuasion experiments with and without defenses, using repeated runs and 95% confidence intervals (CI).

---

## 1. Turns Ablation

**Setup:** We varied persuasion turns (1, 2, 4, 6) under:
- **No Defense** (baseline)
- **Override Defense** (prompt-based factual override)

**Findings (see CI plots):**
- **No Defense:** PSR averaged **~9–12%**, peaking near **12% at 2 turns**. CIs are wide, reflecting small dataset size, but persuasion is consistently nonzero.
- **Override:** PSR stayed **0% across all turns** with tight CIs at 0.
- **RA:** 0% in all cases (no persistence observed).
- **LocAcc:** 100% across conditions, showing defenses did not harm unrelated knowledge.

---

## 2. Order Ablation

**Setup:** We shuffled persuasion strategy order (10+ random shuffles).

**Findings:**
- **No Defense:** Order strongly affects success. Some shuffles averaged **~54% PSR**, while others stayed near 9–12%. CIs confirm substantial variance.
- **Override:** PSR remained **0% in all shuffles**, eliminating order sensitivity.
- **LocAcc:** Again perfect at 100%.

---

## 3. Key Observations

1. **Attacks succeed modestly** without defenses, but can spike high under favorable orderings.
2. **Override defense is consistently effective**, driving PSR to zero while preserving LocAcc.
3. **RA probing was too weak** here; persistence was always 0%. Richer paraphrases or LLM-judge scoring are needed.
4. **Strategy order matters greatly** under no defense, but defenses collapse this variance.

---

## 4. Figures (95% CI)

- **PSR vs Turns**  
  ![](notebooks/results/plots/psr_vs_turns_ci.png)

- **RA vs Turns**  
  ![](notebooks/results/plots/ra_vs_turns_ci.png)

- **LocAcc vs Turns**  
  ![](notebooks/results/plots/locacc_vs_turns_ci.png)

- **Order Ablation: PSR by Shuffle**  
  ![](notebooks/results/plots/order_ablation_psr_ci.png)
