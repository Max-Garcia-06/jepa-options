# Vol-JEPA Round 1 — Design

Date: 2026-11-23. Owner: Max. Status: draft for approval.

## 1. Intent

Test whether a JEPA (joint-embedding predictive architecture) representation of
the daily SPX volatility state carries tradeable information that simple
baselines don't, across three vol strategies, and come back with findings to
refine or pivot. Volatility is latent; the observed vol indexes and realized
vol are its noisy projections — the setup JEPA is built for.

Context: `ca-jepa` (2026-07) closed "JEPA on daily OHLCV for cross-sectional
returns" with a pre-registered negative result (pretraining *removed* signal vs
raw features). This round tests a different hypothesis — vol state, not return
direction — under the same discipline.

## 2. Success definition (agreed 2026-11-23)

- **Tier 1 — research result:** JEPA embeddings beat raw-feature and PCA
  controls on the probe targets. A clean negative result is still a valid,
  write-up-able outcome.
- **Tier 2 — strategy:** a strategy beats "always short vol" on Sharpe **and**
  drawdown, net of costs, through stress periods. If a baseline wins, that's a
  useful strategy but a JEPA failure, and is reported as such.
- **Tier 3 — paper:** a strategy tracks its backtest ~2 months on Alpaca paper.
  **Out of scope for round 1.** Only proposed to Max if Tier 2 passes.

## 3. Goals, non-goals, anti-goals

Goals: one shared JEPA encoder; three strategies probed from it; identical-head
baselines; pre-registered gate; one test-set look; findings report.

Non-goals: full IV surface (round 2, needs WRDS), single stocks, intraday,
live or paper order placement, hyperparameter search on test data.

Anti-goals (failure even if code works):
- any look-ahead leakage (ca-jepa r01's backtest was invalid for this reason);
- reporting validation-tuned numbers as out-of-sample;
- adding/changing strategies, thresholds, or criteria after seeing validation
  results without labeling them exploratory;
- a strategy that "wins" only by being short vol through calm periods and
  blowing up in a stress window.

## 4. Data (verified 2026-11-23)

| Series | Source | Starts |
|---|---|---|
| VIX | Cboe CSV | 1990 |
| VIX6M | Cboe CSV | 2008-01-02 |
| VVIX | Cboe CSV | 2006-03-06 |
| SKEW | Cboe CSV | 1990 |
| VIX3M | Cboe CSV | 2009-09-18 |
| VIX9D | Cboe CSV | 2011-01-04 |
| SPX close | yfinance `^GSPC` | 1990 |
| SPY option bars | Alpaca historical | ~2024-02 |

Cboe URL pattern: `https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYM}_History.csv`
(needs `-L`, redirects). Cache raw downloads under `data/raw/` (gitignored).

**Feature set (core, from 2008-01):** SPX log return; realized vol at 5/10/21
days (close-to-close, annualized); VIX; VIX6M; VVIX; SKEW; derived VIX/VIX6M
ratio and VIX − RV21 spread.
**Extended set (from 2011-01):** core + VIX9D, VIX3M, and their ratios to VIX.
Primary run uses **core** so 2008 is in-sample; extended is a pre-registered
secondary run (see §8).

Normalization: z-score with **train-split** mean/std only.

## 5. Splits

Chronological, 21-trading-day embargo between splits (= target horizon):
- train: 2008-01 → 2016-12 (includes 2008, 2010, 2011)
- val: 2017-01 → 2020-12 (includes Feb 2018 Volmageddon, Mar 2020)
- test: 2021-01 → latest (includes 2022 bear, Apr 2025, current Nov 2026 stress)

2008 is in train, so "survives 2008" is in-sample only; the stress criteria use
val/test events.

**Inner split (for all tuning):** train is split into inner-fit 2008-01 →
2014-12 and inner-select 2015-01 → 2016-12 (21-day embargo). Probe
regularization, S2/S3 thresholds, and S3 hedge size are chosen on inner-select,
then probes are refit on full train. **Val is touched only by the gate.**
Before the single test look, probes are refit on train+val with the same
hyperparameters; the encoder stays frozen (pretrained on train only).
*(Builder-check call.)*

**Test end:** last day with a fully realized 21-day forward window (≈ data end
minus 21 trading days; ~late Oct 2026 as of writing). The Nov 2026 stress period
is reported later as a dated addendum once its labels exist, not in the gate.
*(Builder-check call.)*

## 6. Model (Approach A — shared encoder)

- Input: context window of L = 60 trading days of features ending at close t.
- Context encoder: 1-layer GRU (hidden 64) → LayerNorm → Linear → d = 32.
- Target encoder: same architecture, EMA of context encoder, applied to the
  **next** 21 days of features (t+1..t+21).
- Predictor: MLP from context embedding → predicted target embedding.
- Loss: MSE in latent space + VICReg variance/covariance terms on online
  embeddings (unit-test that gradients flow through the regularizer — ca-jepa
  had a zero-gradient regularizer bug).
- **At inference only the context encoder is used, on data ≤ t.** The target
  encoder never touches probe/backtest features (ca-jepa r01 leak).
- Fixed epoch count, pre-registered (val loss under a moving EMA target is not a
  valid selection criterion — ca-jepa lesson). 5 seeds; report median + spread.
- Pretrain on train split only. Frozen thereafter.

## 7. Strategies, targets, probes

Execution timing for all: signal computed at close t, position opened at close
t+1, P&L over t+1..t+22. Daily laddered entries (1/21 notional each day,
overlapping) to avoid start-date luck.

P&L proxy — **short 30-day variance swap**: payoff per unit =
(VIX_entry/100)² − RV²_realized(next 21 days), with cost = entry vol reduced by
1.0 vol point (sensitivity run at 2.0). Validated on real SPY option P&L where
Alpaca data exists (§9).

| | Strategy | Probe target | Probe | Position rule |
|---|---|---|---|---|
| S1 | VRP timing | forward VRP = VIX² − RV²_fwd21 | ridge regression | weight = clip(pred / train-std, 0, 2) |
| S2 | Crash gate | 1 if short-var P&L in worst 10% (threshold from train) | logistic regression | always-short × (1 − p), p = predicted prob |
| S3 | Hedge overlay | reuses S2 probe | — | long SPX + long variance notional h when p > τ (τ set on train) |

Probes: same head, same regularization grid (selected on inner-select, §5), for
every input.

**S3 details (builder-check calls):** hedge notional h ∈ {0.1, 0.2, 0.3} (long
variance notional per unit SPX), chosen on inner-select by Sharpe; the same h is
used for the always-hedged baseline. τ chosen on inner-select from S2 probe
probabilities fit on inner-fit (so τ is set on out-of-sample probabilities).
S3 has no G1 of its own (it shares S2's probe).

**Position marking:** daily mark-to-market of each laddered short-variance
position with the standard approximation
MTM_t = (elapsed/21)·(RV²_accrued − K²) + (remaining/21)·(VIX_t² − K²),
K = entry vol after cost. Stress-window loss = sum of MTM changes inside the
window across **all** live positions, including ones opened before it.

## 8. Baselines (same heads where applicable)

- Raw-feature control: the 60-day window flattened → PCA to d = 32 → same head.
- Last-day raw features → same head.
- HAR-RV (S1): forecast RV_fwd from RV 1/5/22 → VRP = VIX² − forecast².
- Always short vol (S1, S2 benchmark).
- Term-structure rule: short only when VIX < VIX6M (contango).
- S3: always hedged, never hedged.

## 9. Pre-registered gate (fixed before any model output)

Evaluated on **val**. Three strategies → Bonferroni α = 0.05/3.

**Seeds:** the gate is run on seed-averaged probe outputs (5 encoders → 5
probes → mean prediction). Additionally, ≥ 4 of 5 seeds must individually show
a positive G1 difference. Bootstrap CIs are computed on the seed-averaged
series. *(Builder-check call.)*

- **G1 (Tier 1):** JEPA probe beats both raw controls on its metric — S1
  out-of-sample R², S2 Brier score — with block-bootstrap (21-day blocks) CI of
  the difference excluding 0 at α.
- **G2 (Tier 2):** net Sharpe > every comparator's net Sharpe **and** max
  drawdown ≤ the drawdown benchmark, under 1.0 vol-pt cost.
  - S1, S2 comparators: always-short, term-structure rule, HAR (S1), and the
    raw-PCA and last-day probe strategies run through the identical position
    rule. Drawdown benchmark: always-short.
  - S3 comparators: always-hedged, never-hedged, PCA/last-day probe versions.
    Drawdown benchmark: never-hedged.
  - Reported two ways: **strategy pass** (beats always-short / term-structure
    / HAR, or for S3 always/never-hedged) and **JEPA pass** (also beats the
    PCA/last-day versions). Only a JEPA pass counts as a JEPA success,
    per §2. *(Builder-check call.)*
- **G3 (stress):** loss inside each stress window ≤ drawdown benchmark's loss.
  Val windows: 2018-01-26 → 2018-02-28; 2020-02-19 → 2020-04-30.

A strategy that passes G1–G3 on val gets **one** test-set evaluation, with
G1–G3 applied unchanged. Test stress windows: 2022-01-03 → 2022-10-14;
2025-04-01 → 2025-04-30. Extended feature run is reported alongside but cannot
rescue a failed primary.

**Alpaca check (not a gate):** for 2024-02 → test end, rerun the passing
strategy with real SPY ATM straddles from Alpaca daily bars: enter ~30 DTE,
**delta-hedged daily at close** with Black-Scholes delta from the straddle's
implied vol, exit after 21 trading days. Report correlation of daily P&L with
the proxy and sign agreement of cumulative P&L. (An unhedged straddle would not
be comparable to a variance-swap proxy.) *(Builder-check call.)*

## 10. Leakage guards (tests required)

- Shift test: features at t contain nothing dated > t.
- Embargo test: no train target window overlaps val.
- Target-encoder isolation: probe/backtest code path never imports or calls
  the target encoder.
- Positive control: inject a synthetic feature equal to the true forward VRP
  (lagged correctly to be illegal) — pipeline must score it near-perfect, and
  the shift test must flag it.

## 11. Deliverable

`docs/reports/2026-MM-DD--vol-jepa--r01--findings.md`: gate table per
strategy, baseline comparisons, stress-window P&L, Alpaca proxy check, seeds
spread, and a refine/pivot recommendation. Then a short summary to Max.

## 12. Approaches considered

- **A — shared JEPA encoder, three probes (chosen).** One pretraining; cleanly
  isolates "does the representation help"; matches ca-jepa's gate pattern.
- B — separate end-to-end model per strategy. Rejected: 3× tuning, more
  overfitting on ~4,500 days, harder to separate signal from luck.
- C — full IV surface via WRDS. Deferred to round 2 (blocked on registration);
  will reuse this gate.

## 13. Reuse

May borrow from `~/dataProjects/ml/ca-jepa/v2` (encoder/predictor/EMA trainer,
Newey-West, block bootstrap) — copied and adapted, not imported. Fresh repo.

## 14. Left to the builder

Package layout, caching format, plotting, logging, exact VICReg weights and
learning rate (fixed before val is looked at), CPU vs MPS.
