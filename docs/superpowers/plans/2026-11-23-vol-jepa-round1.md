# Vol-JEPA Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a pre-registered test of whether a JEPA embedding of the daily SPX vol state beats raw-feature controls and simple baselines across three vol strategies (VRP timing, crash gate, hedge overlay).

**Architecture:** Package `volj/` in the existing `jepa-options` repo. Free daily data (Cboe CSVs + yfinance `^GSPC`) → feature panel → one GRU JEPA encoder pretrained on train → frozen context-encoder embeddings → linear probes → laddered short-variance-swap P&L proxy → gate on val → at most one test look. A separate Alpaca module checks the proxy against real delta-hedged SPY straddles (2024-02 →).

**Tech Stack:** Python ≥3.11, uv, torch, numpy, pandas, scikit-learn, scipy, yfinance, requests, pyarrow, pytest, alpaca-py (already installed).

**Spec:** `docs/superpowers/specs/2026-11-23-vol-jepa-round1-design.md`

## Global Constraints

- All pre-registered values live in `volj/config.py` and are committed (Task 1) **before** any model output exists. Changing one after val is seen = exploratory, must be labeled in the report.
- Val is touched only by the gate. All tuning uses inner-fit 2008-01 → 2014-12 / inner-select 2015-01 → 2016-12.
- Inference uses the **context encoder only**. Target encoder never leaves `volj/train.py`.
- Signal at close t → position opened at close t+1 → P&L over t+1..t+22 (21 daily returns t+2..t+22).
- Normalization: z-score with train-split mean/std only.
- Unit tests make no network calls (fixtures/monkeypatch). Network only in `volj/data/fetch.py` and `volj/alpaca_check.py`.
- Raw data → `data/raw/`, artifacts → `outputs/r01/` (both gitignored). Never commit `.env`, account numbers, or data files (repo is public).
- Alpaca: `TradingClient(..., paper=True)` / data clients only. No order placement anywhere in this plan.
- Test-set evaluation runs at most once; enforced by a lock file.

## Review Focus

1. Series with different holiday calendars / gaps (VVIX, SKEW miss days SPX has) → align on SPX dates, forward-fill ≤ 2 days, drop the rest, log the count; never silent NaN→0. (Task 3)
2. Cboe returning an HTML redirect page or a changed header instead of CSV (seen 2026-11-23 without `-L`) → fetch raises `ValueError` naming the symbol; cached file is not written. (Task 2)
3. Last 22 dates have no label → excluded from every split's eval dates and from probe fitting, never filled. (Task 4)
4. Too few crash positives in a fit window → logistic probe raises `ValueError` rather than fitting a constant. (Task 8)
5. Re-running `--stage test` (or running it before val passed) → refuses with a clear error; lock file untouched. (Task 9)

---

### Task 1: Dependencies, package skeleton, pre-registered config

**Files:**
- Modify: `pyproject.toml` (deps, pytest config, package find)
- Create: `volj/__init__.py`, `volj/data/__init__.py`, `volj/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `volj.config` constants below, used by every later task.

- [ ] **Step 1: Add deps**

Run: `uv add torch numpy pandas scikit-learn scipy yfinance requests pyarrow && uv add --dev pytest`
Add to `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = ["."]`. Also create `.python-version` containing `3.11` and run `uv sync` (venv is currently 3.13 because `uv init --bare` wrote no pin).
Expected: `uv run python -c "import torch, sklearn, yfinance"` exits 0.

- [ ] **Step 2: Write failing test `tests/test_config.py::test_preregistered_values`**

```python
from volj import config as c

def test_preregistered_values():
    assert (c.H, c.L, c.D, c.HIDDEN) == (21, 60, 32, 64)
    assert c.SPLITS["train"] == ("2008-01-02", "2016-12-30")
    assert c.SPLITS["val"] == ("2017-01-03", "2020-12-31")
    assert c.SPLITS["test"] == ("2021-01-04", None)
    assert c.INNER["inner_fit"] == ("2008-01-02", "2014-12-31")
    assert c.INNER["inner_select"] == ("2015-01-02", "2016-12-30")
    assert c.COST_VOLPT == 1.0 and c.COST_SENS == 2.0
    assert c.SEEDS == (0, 1, 2, 3, 4) and c.MIN_SEEDS_POSITIVE == 4
    assert (c.EPOCHS, c.LR, c.WD, c.BATCH, c.EMA) == (30, 1e-3, 1e-4, 128, 0.996)
    assert (c.W_SIM, c.W_VAR, c.W_COV) == (25.0, 25.0, 1.0)
    assert c.CRASH_Q == 0.10 and c.S1_CLIP == (0.0, 2.0)
    assert c.S3_H_GRID == (0.1, 0.2, 0.3)
    assert c.S3_TAU_GRID == (0.10, 0.15, 0.20, 0.30, 0.40)
    assert abs(c.ALPHA - 0.05 / 3) < 1e-12
    assert (c.N_BOOT, c.BLOCK, c.BOOT_SEED) == (2000, 21, 12345)
    assert c.STRESS_VAL == [("2018-01-26", "2018-02-28"), ("2020-02-19", "2020-04-30")]
    assert c.STRESS_TEST == [("2022-01-03", "2022-10-14"), ("2025-04-01", "2025-04-30")]
    assert len(c.RIDGE_ALPHAS) == 13 and len(c.LOGIT_CS) == 13
    assert c.FEATURE_SETS["core"] == ["ret", "rv5", "rv10", "rv21", "vix", "vix6m",
                                      "vvix", "skew", "vix_vix6m", "vix_rv21"]
    assert c.FEATURE_SETS["extended"] == c.FEATURE_SETS["core"] + [
        "vix9d", "vix3m", "vix9d_vix", "vix3m_vix"]
    assert (c.ALPACA_CHECK_START, c.RATE, c.DIV_YIELD) == ("2024-02-01", 0.04, 0.013)
    assert c.MIN_CRASH_POSITIVES == 10 and c.MAX_FFILL == 2
```

- [ ] **Step 3: Run** `uv run pytest tests/test_config.py -v` → FAIL (module missing).
- [ ] **Step 4: Implement `volj/config.py`** with exactly those values (`RIDGE_ALPHAS = tuple(np.logspace(-3, 3, 13))`, same for `LOGIT_CS`). Comment at top: "Pre-registered 2026-11-23. Do not edit after val results exist."
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `feat: pre-registered round-1 config and deps` (include `uv.lock`, `.python-version`).

---

### Task 2: Data fetch with cache and format validation

**Files:**
- Create: `volj/data/fetch.py`
- Test: `tests/test_fetch.py` (fixtures inline as strings)

**Interfaces:**
- Produces:
  - `fetch_cboe(sym: str, cache_dir: Path, refresh: bool = False) -> pd.Series` — close, `DatetimeIndex` named `date`, series name = `sym.lower()`, raw index points (e.g. 17.24).
  - `fetch_spx(cache_dir: Path, refresh: bool = False) -> pd.Series` — `^GSPC` daily close, name `spx`, from 2005-01-01.
  - `load_raw(cache_dir: Path, syms: Sequence[str]) -> dict[str, pd.Series]` — keys `spx` + lowercased syms.
  - `CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"`

- [ ] **Step 1: Failing tests**
  - `test_parse_ohlc_layout`: text `"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n01/03/1990,18.19,18.19,18.19,18.19\n"` → series `[17.24, 18.19]`, index `[pd.Timestamp("1990-01-02"), pd.Timestamp("1990-01-03")]`, name `vix`.
  - `test_parse_single_column_layout`: `"DATE,VVIX\n03/06/2006,71.73\n"` → `[71.73]`, name `vvix`.
  - `test_html_response_raises`: monkeypatch `requests.get` to return text `"<head><title>307 Temporary Redirect</title></head>"` → `pytest.raises(ValueError, match="VIX")`, and no file in `cache_dir`.
  - `test_cache_hit_skips_network`: write a valid CSV to `cache_dir/VIX.csv`; monkeypatch `requests.get` to raise; `fetch_cboe("VIX", cache_dir)` returns parsed series.
- [ ] **Step 2: Run** `uv run pytest tests/test_fetch.py -v` → FAIL.
- [ ] **Step 3: Implement.** `requests.get(url, timeout=30, allow_redirects=True)`; validate first line starts with `DATE,`; take `CLOSE` if present else the second column; parse `%m/%d/%Y`; write cache only after validation. `fetch_spx` uses `yfinance.download("^GSPC", start="2005-01-01", auto_adjust=False)["Close"]`, cached as parquet.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Manual check** `uv run python -c "from pathlib import Path; from volj.data.fetch import load_raw; r=load_raw(Path('data/raw'), ['VIX','VIX6M','VVIX','SKEW','VIX3M','VIX9D']); print({k:(v.index[0].date(), v.index[-1].date(), len(v)) for k,v in r.items()})"` → VIX6M first date 2008-01-02, VIX9D 2011-01-04.
- [ ] **Step 6: Commit** `feat: Cboe/SPX fetch with cache and validation`.

---

### Task 3: Feature panel, alignment, scaler, look-ahead check

**Files:**
- Create: `volj/data/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `load_raw` output dict, `config.FEATURE_SETS`, `config.MAX_FFILL`.
- Produces:
  - `build_panel(raw: dict[str, pd.Series], feature_set: str) -> pd.DataFrame` — index = SPX dates, columns = `FEATURE_SETS[feature_set]` in order. Definitions: `ret` = log return; `rvN` = `sqrt(252/N * sum of last N squared log returns)`; `vix, vix6m, vvix, skew, vix9d, vix3m` = level/100 (vvix, skew also /100); `vix_vix6m = vix/vix6m`; `vix_rv21 = vix - rv21`; `vix9d_vix = vix9d/vix`; `vix3m_vix = vix3m/vix`. Non-SPX series reindexed to SPX dates with `ffill(limit=MAX_FFILL)`; rows with any NaN dropped; logs dropped-row count.
  - `fit_scaler(panel: pd.DataFrame, fit_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]` (mean, std).
  - `apply_scaler(panel, mean, std) -> pd.DataFrame`.
  - `assert_no_lookahead(builder: Callable[[dict], pd.DataFrame], raw: dict, check_dates: Sequence[pd.Timestamp]) -> None` — for each date, rebuild from raw truncated to `<= date` and assert the row at `date` equals the full-build row (atol 1e-12); raises `AssertionError` naming the column otherwise.

- [ ] **Step 1: Failing tests** (synthetic raw: 300 business days, SPX random walk seed 0, others constant + noise)
  - `test_columns_and_order`: `list(build_panel(raw,"core").columns) == FEATURE_SETS["core"]`.
  - `test_rv21_value`: SPX log returns alternating ±a with `a = 0.20/np.sqrt(252)` → `rv21` ≈ 0.20 (abs 1e-9) on every row after warmup.
  - `test_ffill_limit`: remove 3 consecutive VVIX dates → first two filled, third row dropped (row count = full − 1).
  - `test_no_lookahead_passes`: `assert_no_lookahead(lambda r: build_panel(r,"core"), raw, raw["spx"].index[100:110])` doesn't raise.
  - `test_no_lookahead_catches_leak`: builder adding column `leak = ret.shift(-1)` → `pytest.raises(AssertionError, match="leak")`.
  - `test_scaler_uses_fit_dates_only`: mean from `fit_scaler(panel, panel.index[:100])` equals `panel.iloc[:100].mean()`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: feature panel with alignment and look-ahead check`.

---

### Task 4: Targets and splits

**Files:**
- Create: `volj/targets.py`, `volj/splits.py`
- Test: `tests/test_targets.py`, `tests/test_splits.py`

**Interfaces:**
- Produces (`volj/targets.py`, all indexed by signal date t, NaN where the window is unavailable):
  - `forward_rv2(spx: pd.Series, h: int = H) -> pd.Series` = `252/h * sum(r[t+2..t+h+1]**2)`, r = log returns.
  - `entry_k2(vix_pts: pd.Series, cost_volpt: float) -> pd.Series` = `((vix_pts.shift(-1) - cost_volpt)/100)**2`.
  - `vrp_target(spx, vix_pts, cost_volpt) -> pd.Series` = `entry_k2 - forward_rv2` (equals short-var payoff per unit).
  - `crash_threshold(vrp: pd.Series, fit_dates) -> float` = `vrp.loc[fit_dates].quantile(CRASH_Q)`.
  - `crash_label(vrp, threshold) -> pd.Series` = `1.0 where vrp <= threshold else 0.0`, NaN kept.
- Produces (`volj/splits.py`):
  - `split_dates(index: pd.DatetimeIndex, name: str, h: int = H) -> pd.DatetimeIndex` — `name` in `SPLITS` ∪ `INNER`; dates t in [start, end] whose label window ends inside the split: `pos(t) + h + 1 <= pos(last index date <= end)`. `None` end = last index date. (This is the embargo.)
  - `test_end(index, h=H) -> pd.Timestamp` = last date of `split_dates(index, "test")`.

- [ ] **Step 1: Failing tests**
  - `test_forward_rv2_alignment`: returns ±a (a = 0.20/√252) → `forward_rv2` = 0.04 (abs 1e-12); last 22 values NaN.
  - `test_forward_rv2_uses_t_plus_2`: one log return of 0.05 at position 30, zeros elsewhere → window of t is returns t+2..t+22, so `forward_rv2.iloc[8] == forward_rv2.iloc[28] == 252/21*0.05**2` (abs 1e-12), and `forward_rv2.iloc[7] == forward_rv2.iloc[29] == 0`.
  - `test_vrp_cost`: VIX constant 20, RV 0.20 → `vrp_target(..., cost_volpt=1.0)` = `0.19**2 - 0.04` = -0.0039 (abs 1e-12).
  - `test_crash_label_nan_preserved`: NaN vrp → NaN label.
  - `test_split_embargo`: on business-day index 2007-06-01 → 2026-11-20, `split_dates(idx,"train")[-1]` is 22 positions before the last index date ≤ 2016-12-30; val first date ≥ 2017-01-03.
  - `test_unlabeled_tail_excluded`: `test_end(idx)` is 22 positions before `idx[-1]`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: VRP/crash targets and embargoed splits`.

---

### Task 5: Variance-swap ladder P&L and metrics

**Files:**
- Create: `volj/pnl.py`
- Test: `tests/test_pnl.py`

**Interfaces:**
- Consumes: SPX close, VIX points.
- Produces:
  - `short_var_ladder_pnl(weights: pd.Series, spx: pd.Series, vix_pts: pd.Series, cost_volpt: float, h: int = H) -> pd.Series` — daily P&L indexed by SPX dates. Weight w_t at signal date t opens notional `w_t/h` at close t+1 (entry e), strike `K² = ((VIX_e − cost)/100)²`. Long-var mark on day s, n = s − e ∈ [0, h]: `MTM = (n/h)(RV²_acc − K²) + ((h−n)/h)((VIX_s/100)² − K²)`, `RV²_acc = 252/n · Σ r²(e+1..s)` (n = 0 → accrued term 0). Short P&L on day s = −notional · (MTM_s − MTM_{s−1}), with MTM before entry = 0, so the entry day books the cost (VIX_e² − K² > 0). Total short P&L over the life = notional · (K² − RV²). Position closes at n = h. NaN weights = 0.
  - `long_spx_pnl(spx) -> pd.Series` simple daily returns.
  - `sharpe(pnl: pd.Series) -> float` = `mean/std * sqrt(252)`.
  - `max_drawdown(pnl) -> float` — on `pnl.cumsum()`, positive number.
  - `window_loss(pnl, start: str, end: str) -> float` = `-pnl.loc[start:end].sum()`.

- [ ] **Step 1: Failing tests** (h = 2 for hand math where noted)
  - `test_single_position_total_equals_payoff`: one weight 1.0 at t0, VIX constant 20, returns ±a (RV 0.20), cost 1.0, h=21 → `pnl.sum()` = `(1/21) * (0.19**2 - 0.04)` (abs 1e-10).
  - `test_mtm_vix_spike_hits_immediately`: h=2, weight 1 at t0, entry VIX 20, next-day VIX 40, zero returns, cost 0 → P&L on day e+1 = `-(1/2)*((1/2)*(0 - 0.04) + (1/2)*(0.16 - 0.04))` = -0.02.
  - `test_ladder_linear_in_weights`: P&L(2·w) == 2·P&L(w).
  - `test_entry_lag`: weight placed at last date produces zero P&L everywhere (entry beyond data).
  - `test_metrics`: `max_drawdown(pd.Series([1,-3,1,1]))` == 3; `window_loss` on known series; `sharpe` of constant-positive-plus-noise > 0.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** vectorized over positions (loop over entry dates acceptable; ~5k × 21).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: laddered short-variance MTM P&L and metrics`.

---

### Task 6: JEPA model and pretraining

**Files:**
- Create: `volj/model.py`, `volj/train.py`
- Test: `tests/test_model.py`, `tests/test_train.py`

**Interfaces:**
- Produces (`volj/model.py`):
  - `class ContextEncoder(nn.Module)`: `__init__(n_features: int, hidden: int = HIDDEN, d: int = D)`; 1-layer GRU → LayerNorm(hidden) → Linear(hidden, d); `forward(x: (B,T,F)) -> (B,d)`. The target encoder is a second instance of this class.
  - `class Predictor(nn.Module)`: `__init__(d: int = D, hidden: int = 128)`; MLP d→hidden→d, GELU.
  - `vicreg_terms(z: Tensor) -> tuple[Tensor, Tensor]` — variance hinge `mean(relu(1 − sqrt(var + 1e-4)))`, covariance `sum(offdiag(cov)²)/d`.
- Produces (`volj/train.py`):
  - `build_windows(X: np.ndarray, idx: np.ndarray, L: int = L, h: int = H) -> tuple[np.ndarray, np.ndarray]` — for each position t in idx: `ctx = X[t-L+1:t+1]`, `tgt = X[t+1:t+h+1]`; positions without full windows skipped.
  - `pretrain(X: np.ndarray, train_pos: np.ndarray, seed: int, epochs: int = EPOCHS) -> ContextEncoder` — EMA target (`EMA`), AdamW(`LR`, `WD`), batch `BATCH`, loss `W_SIM·MSE(pred(ctx_z), sg(tgt_z)) + W_VAR·var + W_COV·cov` with var/cov on online ctx embeddings. `torch.manual_seed(seed)`, `np.random.seed(seed)`. Returns **only** the context encoder in eval mode, with attribute `loss_history: list[float]` (mean loss per epoch). Only `train_pos` whose target window ends ≤ last train position are used.

- [ ] **Step 1: Failing tests**
  - `test_encoder_shape`: `ContextEncoder(10)(torch.randn(4,60,10)).shape == (4,32)`.
  - `test_vicreg_gradient_flows`: `z = torch.randn(64,32, requires_grad=True)*0.01`; `v,c = vicreg_terms(z)`; `(v+c).backward()`; `z.grad.abs().sum() > 0` (ca-jepa zero-gradient bug).
  - `test_build_windows`: X = `np.arange(100)[:,None]`, idx `[59, 60]`, L=60, h=21 → ctx[0,-1,0]==59, tgt[0,0,0]==60, tgt[0,-1,0]==80; idx 90 skipped.
  - `test_pretrain_deterministic`: tiny X (300×10), epochs=1 → two runs with seed 0 give identical encoder outputs; seed 1 differs.
  - `test_pretrain_returns_context_only`: return type is `ContextEncoder`, and `not hasattr(enc, "target")`.
  - `test_loss_decreases`: synthetic AR(1) features, epochs=5 → final epoch mean loss < first epoch (`pretrain` logs per-epoch loss to `enc.loss_history`, list[float]).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Device: MPS if available else CPU (builder's call).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: GRU JEPA with EMA target and VICReg`.

---

### Task 7: Embeddings, controls, HAR baseline, encoder-isolation test

**Files:**
- Create: `volj/embed.py`, `volj/controls.py`
- Test: `tests/test_embed_controls.py`, `tests/test_isolation.py`

**Interfaces:**
- Consumes: `ContextEncoder`, `build_windows` logic (context half only), `forward_rv2`, `entry_k2`.
- Produces:
  - `embed(encoder: ContextEncoder, X: np.ndarray, dates: pd.DatetimeIndex, at: pd.DatetimeIndex) -> pd.DataFrame` — rows `at`, columns `z0..z31`, from windows ending at each date (data ≤ t).
  - `pca_control(X, dates, at, fit_on: pd.DatetimeIndex, L=L, d=D) -> pd.DataFrame` — flattened L×F windows, PCA(d) fit on `fit_on` windows only, columns `p0..p31`.
  - `lastday_control(panel_scaled: pd.DataFrame, at) -> pd.DataFrame` — the row at t.
  - `har_vrp(spx, vix_pts, fit_on, at, cost_volpt) -> pd.Series` — OLS of `forward_rv2` on `[rv2_1, rv2_5, rv2_22]` (realized variance over last 1/5/22 days, annualized) fit on `fit_on`; returns `entry_k2 − forecast` at `at`.

- [ ] **Step 1: Failing tests**
  - `test_embed_uses_past_only`: embedding at date t unchanged when X rows after t are replaced with noise.
  - `test_pca_fit_on_only`: PCA components identical whether X rows after `fit_on[-1]` are noise or not.
  - `test_har_recovers_linear`: synthetic forward_rv2 = 0.5·rv2_22 + noise → coefficient ≈ 0.5 (abs 0.05).
  - `test_isolation` (`tests/test_isolation.py`): for each of `volj/embed.py, controls.py, probes.py, strategies.py, gate.py, scripts/run_round1.py` (skip missing files), source text contains neither `"target_enc"` nor `"tgt_enc"` nor `"EMA"`; and `volj.train` exposes no public name containing `target`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: embeddings, PCA/last-day/HAR controls, isolation test`.

---

### Task 8: Probes and strategy position rules

**Files:**
- Create: `volj/probes.py`, `volj/strategies.py`
- Test: `tests/test_probes.py`, `tests/test_strategies.py`

**Interfaces:**
- Produces (`volj/probes.py`):
  - `select_param(kind: Literal["ridge","logit"], Xf, yf, Xs, ys) -> float` — ridge: alpha in `RIDGE_ALPHAS` minimizing MSE on select; logit: C in `LOGIT_CS` minimizing Brier on select.
  - `fit_probe(kind, X: pd.DataFrame, y: pd.Series, param: float)` — `StandardScaler` + `Ridge(alpha)` or `LogisticRegression(C=param, max_iter=1000)`; drops rows with NaN y; logit raises `ValueError("too few crash positives")` if positives < `MIN_CRASH_POSITIVES`.
  - `predict(kind, est, X) -> pd.Series` — ridge value or P(class 1).
  - `sq_error(pred, y) -> pd.Series`, `brier(p, y) -> pd.Series` — per-date losses.
- Produces (`volj/strategies.py`), all weights indexed by signal date:
  - `s1_weights(pred_vrp: pd.Series, target_std: float) -> pd.Series` = `clip(pred/target_std, *S1_CLIP)`.
  - `s2_weights(p_crash: pd.Series) -> pd.Series` = `1 − p`.
  - `s3_pnl(p_crash, tau: float, h_notional: float, spx, vix_pts, cost_volpt) -> pd.Series` = `long_spx_pnl(spx) − h_notional · short_var_ladder_pnl(1[p > tau], ...)` (long var = −short var).
  - `always_short(index) -> pd.Series` (1.0), `term_structure(vix, vix6m) -> pd.Series` (1.0 if vix < vix6m else 0.0, at t).
  - `select_s3(p_select, spx, vix_pts, cost) -> tuple[float, float]` — (tau, h) maximizing Sharpe over `S3_TAU_GRID × S3_H_GRID` on inner-select; `always_hedged` uses the chosen h with tau = −1.

- [ ] **Step 1: Failing tests**
  - `test_select_ridge_picks_small_alpha_on_clean_signal`: y = X·β exactly → selected alpha == `RIDGE_ALPHAS[0]`.
  - `test_logit_too_few_positives`: y with 9 ones → `pytest.raises(ValueError, match="too few")`.
  - `test_brier`: `brier(pd.Series([0.2]), pd.Series([1.0])).iloc[0] == 0.64`.
  - `test_s1_clip`: pred `[-1, 0.5, 10]`, std 1 → `[0, 0.5, 2]`.
  - `test_s3_no_hedge_equals_spx`: p all 0, tau 0.5 → `s3_pnl == long_spx_pnl`.
  - `test_term_structure`: vix `[15, 25]`, vix6m `[20, 20]` → `[1, 0]`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: linear probes and strategy rules`.

---

### Task 9: Gate, test lock, positive control

**Files:**
- Create: `volj/gate.py`
- Test: `tests/test_gate.py`, `tests/test_positive_control.py`

**Interfaces:**
- Produces:
  - `block_bootstrap_ci(d: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = BOOT_SEED) -> tuple[float, float]` — circular block bootstrap of `mean(d)`, two-sided (1−alpha) percentile CI.
  - `g1(loss_jepa_avg: pd.Series, loss_jepa_seeds: list[pd.Series], loss_controls: dict[str, pd.Series]) -> dict` — for each control: `d = loss_control − loss_jepa_avg` (aligned dates); pass iff every control's CI lo > 0 **and** for every control ≥ `MIN_SEEDS_POSITIVE` seeds have `mean(loss_control − loss_seed) > 0`. Returns `{"pass": bool, "per_control": {name: {"mean_diff", "ci", "seeds_positive"}}}`.
  - `g2(pnl: pd.Series, comparators: dict[str, pd.Series], dd_bench: pd.Series, strategy_only: set[str]) -> dict` — `{"strategy_pass", "jepa_pass", "sharpe", "max_dd", "comparators": {...}}`; strategy_pass vs names in `strategy_only`, jepa_pass vs all; both also require `max_dd <= max_drawdown(dd_bench)`.
  - `g3(pnl, bench, windows: list[tuple[str,str]]) -> dict` — pass iff `window_loss(pnl) <= window_loss(bench)` for every window.
  - `acquire_test_lock(out_dir: Path) -> None` — raises `RuntimeError("test set already evaluated")` if `out_dir/test_look.lock` exists; raises `RuntimeError("val gate not passed")` unless `out_dir/val_gate.json` exists with at least one strategy `"eligible_for_test": true`; else writes the lock with timestamp + `git rev-parse HEAD`.
  - `write_report(results: dict, path: Path) -> None` — JSON + sibling `.md` table.

- [ ] **Step 1: Failing tests**
  - `test_ci_excludes_zero_for_clear_signal`: `d = 1 + 0.1*randn(1000)` → lo > 0.
  - `test_ci_includes_zero_for_noise`: `d = randn(1000)` (seed 0) → lo < 0 < hi.
  - `test_g1_requires_all_controls`: beats PCA, loses to last-day → pass False.
  - `test_g1_seed_rule`: averaged beats both but only 3/5 seeds positive → pass False.
  - `test_g2_jepa_vs_strategy_pass`: pnl beats always-short but not PCA version → strategy_pass True, jepa_pass False.
  - `test_g3_window`: strategy loses more than bench in one window → False.
  - `test_lock_refuses_second_look` and `test_lock_requires_val_pass` (tmp_path).
  - `test_positive_control` (`tests/test_positive_control.py`): synthetic raw (Task 3 fixture style, 1500 days); add illegal feature `oracle = vrp_target(...)` (future-dated) to the panel; ridge probe fit on first 1000 labeled dates → R² on the next 300 > 0.9; and `assert_no_lookahead` on a builder that includes `oracle` raises `AssertionError`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: pre-registered gate, test lock, positive control`.

---

### Task 10: Round-1 orchestration

**Files:**
- Create: `scripts/run_round1.py`
- Test: `tests/test_run_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces CLI: `uv run scripts/run_round1.py --stage {fetch,pretrain,val,test} [--feature-set core|extended] [--cost 1.0] [--smoke]`. Artifacts under `outputs/r01/<feature_set>/`:
  - `pretrain`: `encoder_seed{k}.pt` for `SEEDS`, `loss_history.json`.
  - `val`: inner-split tuning (probe params, crash threshold from train, S3 tau/h) → refit probes on full train → val predictions per seed + averaged → G1/G2/G3 per strategy → `val_gate.json/.md` with `eligible_for_test` per strategy (G1 ∧ G2.jepa_pass ∧ G3). Also writes cost-sensitivity (`--cost 2.0`) numbers as report-only.
  - `test`: `acquire_test_lock` → refit probes on train+val (same params) → same gates on test with `STRESS_TEST` → `test_gate.json/.md`. Only eligible strategies are evaluated.
  - Every report includes `git rev-parse HEAD` and the config module's file hash.
- `--smoke`: synthetic data (no network), `EPOCHS=1`, `SEEDS=(0,1)`, `MIN_SEEDS_POSITIVE` = 1, output under `tmp`.

- [ ] **Step 1: Failing test** `test_smoke_val_runs`: subprocess `uv run scripts/run_round1.py --stage val --smoke` exits 0 and `val_gate.json` has keys `S1, S2, S3`, each with `g1/g2/g3/eligible_for_test` (S3's `g1` = S2's, recorded as `"shared_with": "S2"`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Comparator sets: S1 `{always_short, term_structure, har, pca, lastday}`, strategy_only `{always_short, term_structure, har}`, dd_bench always_short; S2 same minus har; S3 `{always_hedged, never_hedged, pca, lastday}`, strategy_only `{always_hedged, never_hedged}`, dd_bench never_hedged. PCA/last-day comparators = same probe kind + same position rule on control features. S1 target_std = std of train vrp.
- [ ] **Step 4: Run** `uv run pytest -q` → all PASS.
- [ ] **Step 5: Commit** `feat: round-1 pipeline orchestration`.

---

### Task 11: Alpaca real-straddle proxy check

**Files:**
- Create: `volj/alpaca_check.py`
- Test: `tests/test_alpaca_check.py` (pure-function tests only)

**Interfaces:**
- Produces:
  - `bs_price(S, K, T, r, q, sigma, kind: Literal["call","put"]) -> float`, `bs_delta(...) -> float`.
  - `straddle_iv(price, S, K, T, r=RATE, q=DIV_YIELD) -> float` (brentq on [0.01, 5]).
  - `hedged_short_straddle_pnl(call_close: pd.Series, put_close: pd.Series, spy_close: pd.Series, K: float, expiry: pd.Timestamp, r=RATE, q=DIV_YIELD, hold: int = H) -> pd.Series` — short 1 straddle at entry close, re-hedge daily at close with `−(Δcall+Δput)` SPY shares using `straddle_iv` of that day, daily P&L for `hold` days.
  - `fetch_straddle(entry: pd.Timestamp, keys) -> tuple[pd.Series, pd.Series, float, pd.Timestamp]` — via alpaca-py `GetOptionContractsRequest` (include expired/inactive) + `OptionBarsRequest` daily: SPY expiry nearest 30 calendar days, strike nearest entry close.
  - `proxy_agreement(real: pd.Series, proxy: pd.Series) -> dict` — `{"daily_corr", "cum_sign_agree", "n_days"}`.
- CLI: `uv run python -m volj.alpaca_check --strategy S1 --out outputs/r01/core/alpaca_check.json` — runs the chosen strategy's weights over `ALPACA_CHECK_START → test_end`, real ladder of hedged straddles (notional scaled like the proxy), compares to proxy.

- [ ] **Step 1: Failing tests**
  - `test_put_call_parity`: `bs_price(call) − bs_price(put) == S·e^{−qT} − K·e^{−rT}` (abs 1e-10) at S=K=500, T=30/365, σ=0.2.
  - `test_straddle_iv_roundtrip`: price from σ=0.25 → `straddle_iv` returns 0.25 (abs 1e-6).
  - `test_hedged_pnl_zero_when_flat`: constant S and option prices (no time decay simulated) → P&L sum == 0.
  - `test_proxy_agreement`: identical series → corr 1.0, sign agree True.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Network only in `fetch_straddle` and CLI; keys from `.env` via `python-dotenv`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: Alpaca delta-hedged straddle proxy check`.

---

### Task 12: Run round 1 and write findings

**Files:**
- Create: `docs/reports/<run-date>--vol-jepa--r01--findings.md`

- [ ] **Step 1:** Confirm `git status` clean and config unchanged since Task 1 commit: `git log --oneline -- volj/config.py` shows one commit.
- [ ] **Step 2:** `uv run scripts/run_round1.py --stage fetch` → raw files present, Task 2 manual-check dates.
- [ ] **Step 3:** `--stage pretrain` (core), then `--feature-set extended`. Expected: 5 encoders each; loss_history decreasing.
- [ ] **Step 4:** `--stage val` (core and extended). Read `val_gate.md`.
- [ ] **Step 5:** If any core strategy `eligible_for_test`: **stop and ask Max** before `--stage test` (single look). If none: no test run.
- [ ] **Step 6:** If a strategy passed val: run Alpaca check on it (else run it for S1 as proxy-validation only, labeled informational).
- [ ] **Step 7:** Write report: gate table per strategy (G1/G2/G3, strategy vs JEPA pass), probe metrics vs controls with CIs, Sharpe/max DD/stress losses vs comparators, cost-2.0 sensitivity, seed spread, extended-set results (labeled secondary), Alpaca agreement, refine/pivot recommendation. Commit report (no data files).
