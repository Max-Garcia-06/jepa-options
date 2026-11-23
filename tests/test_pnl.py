import numpy as np
import pandas as pd
import pytest

from volj.pnl import long_spx_pnl, max_drawdown, sharpe, short_var_ladder_pnl, window_loss
from volj.targets import vrp_target


def _flat_market(n, vix=20.0, a=0.2 / np.sqrt(252)):
    idx = pd.bdate_range("2020-01-01", periods=n)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    spx = pd.Series(100.0 * np.exp(np.cumsum(a * signs)), index=idx)
    return spx, pd.Series(vix, index=idx)


def test_single_position_total_equals_payoff():
    spx, vix = _flat_market(40)
    w = pd.Series(0.0, index=spx.index)
    w.iloc[0] = 1.0
    pnl = short_var_ladder_pnl(w, spx, vix, cost_volpt=1.0, h=21)
    assert pnl.index.equals(spx.index)
    assert pnl.sum() == pytest.approx((1 / 21) * (0.19**2 - 0.04), abs=1e-10)


def test_mtm_vix_spike_hits_immediately():
    idx = pd.bdate_range("2020-01-01", periods=6)
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series([20.0, 20.0, 40.0, 40.0, 40.0, 40.0], index=idx)
    w = pd.Series(0.0, index=idx)
    w.iloc[0] = 1.0
    pnl = short_var_ladder_pnl(w, spx, vix, cost_volpt=0.0, h=2)
    # entry at idx[1] (VIX 20); spike on idx[2]
    assert pnl.iloc[2] == pytest.approx(-0.02, abs=1e-12)
    assert pnl.iloc[0] == 0.0


def test_ladder_linear_in_weights():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=120)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx)
    vix = pd.Series(rng.uniform(12, 30, 120), index=idx)
    w = pd.Series(rng.uniform(0, 2, 120), index=idx)
    p1 = short_var_ladder_pnl(w, spx, vix, 1.0, h=21)
    p2 = short_var_ladder_pnl(2 * w, spx, vix, 1.0, h=21)
    pd.testing.assert_series_equal(p2, 2 * p1)


def test_entry_lag():
    spx, vix = _flat_market(30)
    w = pd.Series(0.0, index=spx.index)
    w.iloc[-1] = 1.0
    pnl = short_var_ladder_pnl(w, spx, vix, 1.0, h=21)
    assert (pnl == 0.0).all()


def test_nan_weights_are_zero_and_no_nan_output():
    spx, vix = _flat_market(40)
    w = pd.Series(np.nan, index=spx.index)
    pnl = short_var_ladder_pnl(w, spx, vix, 1.0, h=21)
    assert not pnl.isna().any() and (pnl == 0.0).all()


def test_partial_life_marked_to_last_day():
    spx, vix = _flat_market(10)
    w = pd.Series(0.0, index=spx.index)
    w.iloc[2] = 1.0  # entry at 3, only 6 more days of 21
    pnl = short_var_ladder_pnl(w, spx, vix, 1.0, h=21)
    assert (pnl.iloc[:3] == 0.0).all()
    assert pnl.iloc[3] == pytest.approx(-(1 / 21) * (0.20**2 - 0.19**2), abs=1e-12)
    assert pnl.iloc[3:].abs().sum() > 0


def test_consistent_with_vrp_target():
    rng = np.random.default_rng(1)
    n, h = 150, 21
    idx = pd.bdate_range("2020-01-01", periods=n)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    vix = pd.Series(rng.uniform(12, 30, n), index=idx)
    w = pd.Series(rng.uniform(0, 2, n), index=idx)
    vrp = vrp_target(spx, vix, 1.0)
    for i in range(0, n - h - 2, 7):
        single = pd.Series(0.0, index=idx)
        single.iloc[i] = w.iloc[i]
        total = short_var_ladder_pnl(single, spx, vix, 1.0, h=h).sum()
        assert total == pytest.approx(w.iloc[i] / h * vrp.iloc[i], abs=1e-10)


def test_metrics():
    assert max_drawdown(pd.Series([1.0, -3.0, 1.0, 1.0])) == pytest.approx(3.0)
    assert max_drawdown(pd.Series([-2.0, 1.0])) == pytest.approx(2.0)
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4))
    assert window_loss(s, "2020-01-02", "2020-01-03") == -5.0
    rng = np.random.default_rng(0)
    assert sharpe(pd.Series(0.1 + rng.normal(0, 0.05, 500))) > 0
    spx = pd.Series([100.0, 110.0, 99.0])
    assert long_spx_pnl(spx).iloc[1:].tolist() == pytest.approx([0.1, -0.1])


def test_drawdown_of_long_spx_finite():
    pnl = long_spx_pnl(pd.Series([100.0, 101.0, 99.0, 102.0, 103.0, 104.0]))
    assert pnl.iloc[0] == 0.0
    assert max_drawdown(pnl) == pytest.approx(1 - 99 / 101, abs=1e-12)


def test_metrics_raise_on_nan():
    s = pd.Series([1.0, np.nan, 2.0])
    for f in (sharpe, max_drawdown):
        with pytest.raises(ValueError):
            f(s)
    with pytest.raises(ValueError):
        window_loss(s, 0, 2)


def test_sharpe_zero_variance_and_empty():
    assert sharpe(pd.Series([0.0] * 5)) == 0.0
    assert sharpe(pd.Series([0.1] * 5)) == 0.0
    assert sharpe(pd.Series([], dtype=float)) == 0.0


def test_missing_vix_on_needed_date_raises():
    spx, vix = _flat_market(40)
    w = pd.Series(0.0, index=spx.index)
    w.iloc[0] = 1.0
    vix = vix.drop(vix.index[5])
    with pytest.raises(ValueError, match="VIX"):
        short_var_ladder_pnl(w, spx, vix, 1.0, h=21)
