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
