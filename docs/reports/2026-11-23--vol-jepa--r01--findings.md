---
type: results-report
date: 2026-11-23
experiment_line: vol-jepa
round: 1
status: final (val gate; no test look)
spec: docs/superpowers/specs/2026-11-23-vol-jepa-round1-design.md
code_rev: fd28039 (clean tree), config pre-registered at e89b787 (unchanged)
artifacts: outputs/r01/{core,extended}/val_gate.{json,md}, val_{preds,weights,pnl}.parquet, params.json; outputs/r01/core/alpaca_check.json (gitignored)
---

# Vol-JEPA / Round 1 / Findings / 2026-11-23

## 1. Bottom line

**No strategy passed the pre-registered validation gate, in either feature set.** No strategy is eligible for the test set, so the test set was never evaluated (no `test_look.lock` exists). It stays untouched for a future pre-registered hypothesis.

- **Tier 1 (representation) failed.** JEPA embeddings beat a PCA summary of the 60-day window for VRP prediction (core set only). They lost to simply using *today's raw features* in every comparison that mattered. For crash prediction, JEPA was worse than both controls.
- **Tier 2 (strategy) failed.** The best JEPA strategy (S1 core, Sharpe 0.47) beat "always short vol" (−0.23) but lost to the HAR baseline (0.58) and the raw last-day probe (1.00).
- This repeats the ca-jepa result on a different hypothesis: **on daily index-level data, JEPA pretraining does not add information beyond the raw inputs.**

## 2. Setup (as pre-registered)

- **Data:** SPX closes (yfinance) plus Cboe VIX, VIX6M, VVIX and SKEW (core). The extended set adds VIX9D and VIX3M. Data runs through 2026-11-20, and the last labeled signal date is 2026-10-21.
- **Splits:** train 2008–2016 (inner tuning: fit 2008–14, select 2015–16), val 2017–2020, test 2021→. There is a 21-day embargo between splits.
- **Model:** GRU JEPA (60-day context, predicts the embedding of the next 21 days), EMA target, VICReg. 5 seeds × 30 epochs, CPU. Training loss fell for every seed (core about 15.8→11.9, extended about 18→10).
- **Probes:** ridge (VRP) and logistic (crash), with the same head for every input: JEPA, PCA-32 of the window, and last-day raw features.
- **P&L:** laddered short 30-day variance-swap approximation, entering at t+1 with a 1 vol-point cost, marked to market daily.
- **Gate:** α = 0.05/3. G1 = JEPA beats both controls (block-bootstrap CI, and at least 4 of 5 seeds beat every control). G2 = Sharpe beats every comparator and drawdown ≤ benchmark. G3 = stress-window loss ≤ benchmark.

## 3. Gate results (val 2017-01-03 → 2020-12-31)

| | G1 vs PCA | G1 vs last-day | G1 | G2 strategy | G2 JEPA | G3 | Eligible |
|---|---|---|---|---|---|---|---|
| **core S1** VRP timing | **+0.00074** CI (0.00012, 0.0016), 5/5 seeds | −0.00017 CI (−0.0014, 0.0007), 0/5 | ✗ | ✗ | ✗ | ✓ | ✗ |
| **core S2** crash gate | −0.0025 | −0.0005 | ✗ | ✗ | ✗ | ✓ | ✗ |
| **core S3** hedge | (shared with S2) | | ✗ | ✗ | ✗ | ✓ | ✗ |
| **ext S1** | −0.00024 | −0.00013 | ✗ | **✓** | ✗ | ✓ | ✗ |
| **ext S2** | −0.040 (sig. worse) | −0.036 (sig. worse) | ✗ | ✗ | ✗ | ✓ | ✗ |
| **ext S3** | (shared) | | ✗ | ✗ | ✗ | ✓ | ✗ |

G1 differences are control loss minus JEPA loss, so positive means JEPA is better.

### Val Sharpe (net of 1 vol-pt cost)

| Strategy | JEPA | always short | term structure | HAR | PCA probe | last-day probe |
|---|---|---|---|---|---|---|
| S1 core | 0.47 | −0.23 | −0.09 | 0.58 | 0.36 | **1.00** |
| S1 extended | 0.04 | −0.23 | −0.09 | −0.06 | 0.44 | 0.11 |
| S2 core | −0.22 | −0.23 | −0.09 | – | −0.23 | −0.25 |

| S3 hedge overlay | JEPA | always hedged | never hedged | PCA | last-day |
|---|---|---|---|---|---|
| core | 0.735 | 0.77 (h=0.1) | 0.735 | 0.76 | 0.73 |
| extended | 0.70 | 0.82 (h=0.3) | 0.735 | 0.75 | 0.735 |

## 4. What the numbers say

1. **The representation doesn't beat the raw inputs.** The only significant JEPA win is against PCA of the 60-day window for VRP (core). PCA is a weak control. Today's raw features (VIX level, term-structure ratio, realized vol) predict VRP at least as well as the JEPA embedding, with 0/5 seeds beating them. For crash prediction, JEPA is worse than both controls, significantly so in the extended set.
2. **The probes found almost nothing to learn.** Every probe, for every input, chose the strongest regularization in its pre-registered grid (ridge α = 1000, logistic C ≈ 0.001) on the 2015–16 tuning data. Predictions are close to constant. As a result:
   - S1 JEPA sat mostly out of the market (mean weight 0.085).
   - S2 is roughly "always short × 0.9" (mean weight 0.90, Sharpe −0.22 vs −0.23).
   - S3's crash probability never crossed τ = 0.3, so **S3 never hedged** and is identical to never-hedged.
3. **G3 passing is not informative.** Every strategy passed the stress test because it held less short-vol exposure than always-short. Low exposure looks safe in a crash. It is not skill.
4. **The validation period was hostile to short vol.** Always-short lost money over 2017–2020 (Sharpe −0.23) because of Feb 2018 and Mar 2020. Anything that reduced exposure looks better than the benchmark.

## 5. Real-option check of the P&L approximation (model-free)

I ran this with always-short weights only, because the window (2024-02 → 2026-10-21) falls inside the held-out test period. No model output was used. Each position is a ~30-DTE ATM SPY straddle from Alpaca daily bars, delta-hedged daily and held 21 days.

| daily corr (real vs approximation) | cumulative sign agrees | OLS beta | entries used / skipped | days |
|---|---|---|---|---|
| **0.82** | yes | 0.54 | 372 / 311 | 698 |

- **The approximation tracks real option P&L direction well.** Real P&L is about half the approximation's size, so the absolute Sharpe and drawdown figures are scaled, but the rankings should hold.
- **Coverage caveat:** 46% of entries were skipped. The causes were missing option bars on some days and stale last-trade prices below their minimum possible value. Skips are spread across 2024, 2025 and 2026 (119 / 114 / 78).

## 6. Exploratory observations (not pre-registered; hypotheses only)

- The **last-day raw-feature VRP probe** had the best val Sharpe seen (1.00). It is a plain linear model on today's VIX, term structure and realized vol. It was a *comparator*, chosen with the same heavy regularization. It is the most promising lead, but a clean claim needs its own pre-registration and the untouched test set.
- The always-hedged comparator beat never-hedged on Sharpe in val (0.77–0.82 vs 0.735). That is about the hedge carry during 2017–2020, not about any signal.

## 7. Known defects and caveats

- **The cost-sensitivity block came out empty** (`cost_sensitivity.strategies = {}`). This is a pipeline bug in a report-only section and doesn't change any verdict. Because nothing passed at 1.0 vol pt, a 2.0 vol-pt cost can only lower short-vol P&L further.
- **Tuning asymmetry (Ruling 5):** the JEPA encoder was pretrained on all of train, including the 2015–16 select years. PCA for tuning was fit only on 2008–14. This favors JEPA, and it still lost.
- **Different crash thresholds:** the core and extended crash labels use different train starts (2008 vs 2011), so their thresholds differ.
- `select_s3` scores P&L only through the last inner-select signal date. That is conservative.
- `params.json` and `val_gate.json` are not hash-pinned for the test stage. This is moot, since no test look was taken.

## 8. Recommendation: pivot away from JEPA on index-level daily vol data

1. **Close "JEPA on daily index vol state."** Tier 1 failed cleanly. Together with ca-jepa, that makes two pre-registered negatives on daily index data. More epochs or architecture tweaks don't address the cause: the raw features already contain the information.
2. **If representation learning stays the goal, change the input, not the model.** Round 2 on the full implied-vol surface needs WRDS or bought data. A surface is high-dimensional and redundant, which is where a learned embedding can plausibly beat raw features. Single-stock earnings vol is another option, with far more samples. Reuse this gate unchanged.
3. **If trading is the goal, pre-register a simple VRP-timing strategy.** Use last-day raw features or HAR as a non-JEPA track, and give it the one test-set look. The gate, P&L engine and Alpaca check are all reusable as built.
4. **Fix the design issues first:** the cost-sensitivity bug; a τ grid that can fire when probabilities sit near the base rate (S3 never hedged); and a regularization grid that doesn't put every input at its edge.
