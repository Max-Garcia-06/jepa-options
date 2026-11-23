import json
import math

import numpy as np
import pandas as pd
import pytest

from volj.alpaca_check import (bs_delta, bs_price, hedged_short_straddle_pnl, occ_symbol,
                               ols_beta, proxy_agreement, require_weights_unlocked,
                               straddle_iv)


def test_put_call_parity():
    S = K = 500.0
    T, r, q, s = 30 / 365, 0.04, 0.013, 0.2
    diff = bs_price(S, K, T, r, q, s, "call") - bs_price(S, K, T, r, q, s, "put")
    assert diff == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)
    dc, dp = bs_delta(S, K, T, r, q, s, "call"), bs_delta(S, K, T, r, q, s, "put")
    assert dc - dp == pytest.approx(math.exp(-q * T), abs=1e-12)


def test_straddle_iv_roundtrip():
    S, K, T, r, q = 500.0, 505.0, 30 / 365, 0.04, 0.013
    p = bs_price(S, K, T, r, q, 0.25, "call") + bs_price(S, K, T, r, q, 0.25, "put")
    assert straddle_iv(p, S, K, T, r, q) == pytest.approx(0.25, abs=1e-6)


def test_hedged_pnl_zero_when_flat():
    idx = pd.bdate_range("2024-03-01", periods=30)
    expiry = idx[25]
    c = pd.Series(8.0, index=idx)
    p = pd.Series(8.0, index=idx)
    spy = pd.Series(500.0, index=idx)
    pnl = hedged_short_straddle_pnl(c, p, spy, 500.0, expiry, hold=21)
    assert len(pnl) == 21
    assert pnl.sum() == pytest.approx(0.0, abs=1e-12)


def test_hedged_pnl_short_straddle_gains_from_decay():
    idx = pd.bdate_range("2024-03-01", periods=10)
    expiry = idx[-1]
    spy = pd.Series(500.0, index=idx)
    c = pd.Series(np.linspace(10, 6, 10), index=idx)
    p = pd.Series(np.linspace(10, 6, 10), index=idx)
    pnl = hedged_short_straddle_pnl(c, p, spy, 500.0, expiry, hold=9)
    assert pnl.sum() == pytest.approx(8.0)


def test_hedged_pnl_missing_option_bar_raises():
    idx = pd.bdate_range("2024-03-01", periods=10)
    spy = pd.Series(500.0, index=idx)
    c = pd.Series(8.0, index=idx).drop(idx[3])
    p = pd.Series(8.0, index=idx)
    with pytest.raises(ValueError, match="missing"):
        hedged_short_straddle_pnl(c, p, spy, 500.0, idx[-1], hold=9)


def test_proxy_agreement():
    idx = pd.bdate_range("2024-03-01", periods=30)
    s = pd.Series(np.random.default_rng(0).normal(size=30), index=idx)
    out = proxy_agreement(s, s)
    assert out["daily_corr"] == pytest.approx(1.0)
    assert out["cum_sign_agree"] is True
    assert out["n_days"] == 30
    assert proxy_agreement(s, -s)["cum_sign_agree"] is False


def test_ols_beta():
    x = pd.Series(np.arange(10.0))
    assert ols_beta(2 * x + 1, x) == pytest.approx(2.0)


def test_occ_symbol():
    assert occ_symbol(pd.Timestamp("2024-03-15"), "call", 510.0) == "SPY240315C00510000"
    assert occ_symbol(pd.Timestamp("2024-03-15"), "put", 510.5) == "SPY240315P00510500"


def test_weights_file_requires_lock(tmp_path):
    with pytest.raises(RuntimeError, match="test_look.lock"):
        require_weights_unlocked(tmp_path)
    (tmp_path / "test_look.lock").write_text("x")
    require_weights_unlocked(tmp_path)


def test_straddle_iv_below_floor_raises():
    S, K, T = 520.0, 500.0, 0.01  # deep ITM: straddle worth less than intrinsic
    with pytest.raises(ValueError, match="below 1% vol value"):
        straddle_iv(15.0, S, K, T)


def _gbm_straddle(seed=0, n=22, sigma=0.2):
    from volj.alpaca_check import RATE, DIV_YIELD
    idx = pd.bdate_range("2024-03-01", periods=n)
    z = np.random.default_rng(seed).normal(size=n - 1)
    S = 500 * np.exp(np.concatenate([[0], np.cumsum(sigma / np.sqrt(252) * z)]))
    expiry = idx[-1] + pd.Timedelta(days=60)
    T = [(expiry - d).days / 365 for d in idx]
    px = {k: pd.Series([bs_price(s, 470.0, t, RATE, DIV_YIELD, sigma, k) for s, t in zip(S, T)],
                       index=idx) for k in ("call", "put")}
    return px["call"], px["put"], pd.Series(S, index=idx), expiry


def test_hedge_sign_reduces_variance():
    c, p, spy, expiry = _gbm_straddle()
    hedged = hedged_short_straddle_pnl(c, p, spy, 470.0, expiry, hold=21)
    unhedged = -(c + p).diff().dropna()
    # same P&L with the hedge sign flipped (shares replaced by their negative)
    wrong = unhedged - (hedged - unhedged)
    assert hedged.std() < 0.3 * unhedged.std()
    assert wrong.std() > 1.5 * unhedged.std()


def test_run_check_counts_subintrinsic_entry_as_skipped(monkeypatch, caplog):
    import volj.alpaca_check as a
    idx = pd.bdate_range("2024-02-01", periods=60)
    spx = pd.Series(5000.0, index=idx) * np.exp(np.arange(60) * 1e-3)
    vix, spy = pd.Series(15.0, index=idx), spx / 10

    def fake(e, keys, spy_close, h):
        d = spy_close.loc[e:].index[:25]
        return pd.Series(0.001, index=d), pd.Series(0.001, index=d), float(round(spy_close[e])), d[-1]

    monkeypatch.setattr(a, "fetch_straddle", fake)
    w = pd.Series(0.0, index=idx)
    w.iloc[:3] = 1.0
    with caplog.at_level("WARNING"):
        real, proxy, n_used, n_skipped = a.run_check(w, spx, vix, spy, None)
    assert (n_used, n_skipped) == (0, 3)
    assert "below 1% vol value" in caplog.text


def test_result_json_nan_to_null():
    from volj.alpaca_check import result_json
    out = json.loads(result_json({"daily_corr": float("nan"), "beta": 1.5}))
    assert out == {"daily_corr": None, "beta": 1.5}
