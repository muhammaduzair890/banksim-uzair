# BankSim Fraud Fidelity Analysis

## What each model is

| Model | Trained on | Purpose |
|-------|-----------|---------|
| **M1** | Fraud rows only (~5k/fold) | Learns pure fraud behavior — amounts, categories, merchant and demographic patterns of fraudsters. |
| **M2** | Fraud + 10% non-fraud, stratified | Learns fraud *in context* of legitimate transactions — gives the model a reference point for "normal". |
| **M3** | Full training set (~400k/fold) | Learns everything — generates fraud at the natural rate (~1.3%), preserving real-world class balance. |

All three synthetic **pools** are filtered to fraud rows only (20,000 each per
fold), so they are compared like-for-like against the **real fraud rows** in
each fold's training split.

---

## The fraud-signature properties

BankSim transactions have a compact schema, and the model operates on:

```
step, age, gender, merchant, category, amount, fraud
```

The fidelity check compares the marginals that carry the fraud signal:

| Property | What it captures | Why it matters |
|---|---|---|
| **category concentration** | Share of fraud in `sportsandtoys`+`health` | Real fraud concentrates here (~52%). |
| **legit-category leak** | Share of fraud in `transportation`+`food` | These dominate legit traffic and ~never carry fraud. |
| **amount mean / median / max** | Transaction-amount distribution | Fraud amounts (~546) are ~14× the overall mean (~38). |
| **merchant concentration, gender, age skew** | Who/where fraud lands | Fraud skews female (~67%) and to age buckets 2–4. |

---

## Results — per fold

Values are reported for each of the 5 folds individually (no averaging). Each
model has its own table; the **Real fraud** table below is the per-fold baseline
to compare every model against.

### Real fraud (baseline)

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| amount mean | 557.6 | 556.6 | 539.6 | 538.2 | 539.0 |
| amount median | 325.0 | 325.0 | 322.0 | 321.4 | 321.9 |
| amount max | 7,432 | 7,666 | 7,666 | 7,666 | 7,666 |
| fraud-core categories (sportsandtoys+health) | 51.1% | 52.0% | 52.7% | 52.0% | 51.5% |
| legit-category leak (transportation+food) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| # distinct categories | 12 | 12 | 12 | 12 | 12 |
| gender == F | 68.6% | 67.6% | 66.7% | 66.4% | 66.3% |
| top age-bucket share | 30.1% | 28.3% | 30.7% | 32.3% | 32.2% |
| # distinct merchants | 29 | 30 | 30 | 30 | 30 |
| top merchant share | 21.8% | 21.8% | 22.7% | 22.6% | 22.6% |

Markers compare each model value against the **same fold's** real-fraud value:
✅ within tolerance · ⚠️ mild drift · ❌ material defect.

### M1 — fraud only

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| amount mean | 587.4 ✅ | 562.2 ✅ | 555.4 ✅ | 533.2 ✅ | 547.7 ✅ |
| amount median | 330.3 ✅ | 328.3 ✅ | 332.1 ✅ | 319.2 ✅ | 325.6 ✅ |
| amount max | 5,737 ✅ | 6,370 ✅ | 6,583 ✅ | 6,572 ✅ | 7,134 ✅ |
| fraud-core categories (sportsandtoys+health) | 53.2% ✅ | 53.3% ✅ | 53.6% ✅ | 55.2% ✅ | 53.4% ✅ |
| legit-category leak (transportation+food) | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ |
| # distinct categories | 12 ✅ | 12 ✅ | 12 ✅ | 12 ✅ | 12 ✅ |
| gender == F | 68.7% ✅ | 66.2% ✅ | 66.0% ✅ | 65.3% ✅ | 66.1% ✅ |
| top age-bucket share | 30.1% ✅ | 27.9% ✅ | 29.9% ✅ | 31.7% ✅ | 32.0% ✅ |
| # distinct merchants | 28 ✅ | 29 ✅ | 29 ✅ | 30 ✅ | 30 ✅ |
| top merchant share | 21.3% ✅ | 21.5% ✅ | 23.9% ✅ | 23.9% ✅ | 24.0% ✅ |

### M2 — fraud + 10% non-fraud

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| amount mean | 519.1 ✅ | 560.3 ✅ | 565.2 ✅ | 574.4 ✅ | 632.8 ✅ |
| amount median | 276.6 ✅ | 298.0 ✅ | 288.4 ✅ | 292.0 ✅ | 303.7 ✅ |
| amount max | 5,771 ✅ | 6,532 ✅ | 6,532 ✅ | 6,649 ✅ | 6,888 ✅ |
| fraud-core categories (sportsandtoys+health) | 52.1% ✅ | 49.3% ✅ | 53.9% ✅ | 53.5% ✅ | 55.5% ✅ |
| legit-category leak (transportation+food) | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ |
| # distinct categories | 12 ✅ | 12 ✅ | 12 ✅ | 12 ✅ | 12 ✅ |
| gender == F | 64.8% ✅ | 66.1% ✅ | 64.8% ✅ | 60.2% ⚠️ | 60.0% ⚠️ |
| top age-bucket share | 29.5% ✅ | 27.5% ✅ | 31.0% ✅ | 32.8% ✅ | 32.0% ✅ |
| # distinct merchants | 43 ⚠️ | 46 ⚠️ | 44 ⚠️ | 45 ⚠️ | 39 ⚠️ |
| top merchant share | 23.9% ✅ | 19.9% ✅ | 22.8% ✅ | 22.8% ✅ | 22.9% ✅ |

### M3 — full training set

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| amount mean | **1,134 ❌** | **1,762 ❌** | **1,814 ❌** | **1,783 ❌** | **1,590 ❌** |
| amount median | 170.3 ⚠️ | 436.6 ⚠️ | 397.6 ✅ | 268.6 ✅ | 204.6 ⚠️ |
| amount max | 5,770 ✅ | 6,370 ✅ | 6,591 ✅ | 6,649 ✅ | 6,679 ✅ |
| fraud-core categories (sportsandtoys+health) | 51.2% ✅ | 54.7% ✅ | 54.3% ✅ | 49.3% ✅ | 51.8% ✅ |
| legit-category leak (transportation+food) | 0.1% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ | 0.0% ✅ |
| # distinct categories | 15 ⚠️ | 14 ⚠️ | 13 ✅ | 13 ✅ | 13 ✅ |
| gender == F | 65.7% ✅ | 64.4% ✅ | 64.3% ✅ | 63.8% ✅ | 63.8% ✅ |
| top age-bucket share | 26.2% ✅ | 29.9% ✅ | 30.8% ✅ | 30.5% ✅ | 30.8% ✅ |
| # distinct merchants | 48 ⚠️ | 46 ⚠️ | 45 ⚠️ | 47 ⚠️ | 42 ⚠️ |
| top merchant share | 20.4% ✅ | 21.7% ✅ | 22.1% ✅ | 21.4% ✅ | 23.1% ✅ |

---

## Interpretation

The **BankSim models reproduce real fraud well**. Two genuine defects stand out:

1. **M3 amount mean is 2–3× inflated in every fold** (1,134–1,814 vs a real
   ~538–558), while its *median* stays in the right neighborhood. The mean and
   median disagree across all folds, so M3 is generating a heavy upper-tail of
   large fraud amounts that real fraud doesn't have, and it is the single most
   important fidelity problem. M3 learns from the full (mostly legitimate)
   distribution, so its conditional amount distribution for the rare fraud class
   is the least well-anchored.

2. **M2 and M3 leak into extra merchants and categories in every fold** (M2: 39–46
   distinct merchants, M3: 42–48, vs 29–30 in real fraud; M3 invents 13–15
   categories vs 12). They place fraud at merchants/categories that real fraud
   never touches. M1, trained on fraud only, stays tight to the real support
   (28–30 merchants, 12 categories) across all folds.

**Takeaways**

- **M1 is the highest-fidelity pool** — near-perfect on every marginal.
- **M2 is a close second** — slightly broader merchant support, otherwise faithful.
- **M3 preserves natural class balance but pays for it in amount fidelity** —
  prefer it when realistic prevalence matters more than per-row amount accuracy;
  otherwise prefer M1/M2.
- **No model leaks fraud into legit-dominant categories** (transportation/food
  stay at 0%), which is the most important structural check — and all pass.

---
