import numpy as np
import pandas as pd
import pytest

from volj.pnl import long_spx_pnl, short_var_ladder_pnl
from volj.strategies import (always_hedged, always_short, s1_weights, s2_weights, s3_pnl,
                             select_s3, term_structure)


@pytest.fixture
def mkt():
    n = 120
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    spx = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    vix = pd.Series(16 + rng.normal(0, 1, n), index=idx)
    return spx, vix


def test_s1_clip():
    out = s1_weights(pd.Series([-1.0, 0.5, 10.0]), 1.0)
    assert out.tolist() == [0.0, 0.5, 2.0]


def test_s2():
    assert s2_weights(pd.Series([0.25])).iloc[0] == 0.75


def test_s3_no_hedge_equals_spx(mkt):
    spx, vix = mkt
    p = pd.Series(0.0, index=spx.index)
    pd.testing.assert_series_equal(s3_pnl(p, 0.5, 0.2, spx, vix, 1.0), long_spx_pnl(spx))


def test_s3_hedge_is_minus_ladder_and_always_hedged(mkt):
    spx, vix = mkt
    p = pd.Series(1.0, index=spx.index)
    lad = short_var_ladder_pnl(pd.Series(1.0, index=spx.index), spx, vix, 1.0)
    expect = long_spx_pnl(spx) - 0.2 * lad
    pd.testing.assert_series_equal(s3_pnl(p, 0.5, 0.2, spx, vix, 1.0), expect)
    pd.testing.assert_series_equal(always_hedged(spx.index, 0.2, spx, vix, 1.0), expect)


def test_term_structure():
    out = term_structure(pd.Series([15.0, 25.0]), pd.Series([20.0, 20.0]))
    assert out.tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        term_structure(pd.Series([np.nan]), pd.Series([20.0]))


def test_always_short():
    assert always_short(pd.RangeIndex(3)).tolist() == [1.0] * 3


def test_select_s3_in_grid(mkt):
    spx, vix = mkt
    rng = np.random.default_rng(1)
    p = pd.Series(rng.uniform(0, 0.5, 60), index=spx.index[:60])
    tau, h = select_s3(p, spx, vix, 1.0)
    from volj.config import S3_H_GRID, S3_TAU_GRID
    assert tau in S3_TAU_GRID and h in S3_H_GRID


def _spike_market(spike_at, n=160):
    """Calm market (VIX 20, ~5% realized vol) with a crash at spike_at: SPX -3%/day for
    5 days, then +-3% swings; VIX jumps to 45 and stays there."""
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.003, n)
    vix = np.full(n, 20.0)
    r[spike_at:spike_at + 5] = -0.03
    r[spike_at + 5:] = 0.03 * (-1.0) ** np.arange(n - spike_at - 5)
    vix[spike_at:] = 45.0
    return pd.Series(2000 * np.exp(np.cumsum(r)), index=idx), pd.Series(vix, index=idx)


def test_select_s3_ignores_data_after_window():
    from volj.config import S3_H_GRID, S3_TAU_GRID
    spx, vix = _spike_market(spike_at=62)
    p = pd.Series(0.0, index=spx.index[:60])
    p.iloc[30:] = 0.35  # hedging loses in-window; it pays only after the window (spike at 62)
    hi = p.index[-1]
    got = select_s3(p, spx, vix, 1.0)
    truncated = select_s3(p, spx.loc[:hi], vix.loc[:hi], 1.0)
    assert got == truncated == (0.4, 0.1)  # in-window: any hedge loses
    # the post-window spike would change the choice if it were not excluded
    from volj.pnl import sharpe
    full = {(t, h): sharpe(s3_pnl(p, t, h, spx, vix, 1.0))
            for t in S3_TAU_GRID for h in S3_H_GRID}
    assert max(full, key=full.get) == (0.1, 0.3)


def test_select_s3_picks_known_best_pair():
    spx, vix = _spike_market(spike_at=45)
    p = pd.Series(0.0, index=spx.index[:80])
    p.iloc[:15] = 0.25                         # spurious mid-level hedges in calm, expire pre-spike
    p.iloc[40:44] = 0.35                       # high just before the in-window spike
    # tau 0.30 hedges only the 0.35 days; lower taus also hedge (losing) calm days
    assert select_s3(p, spx, vix, 1.0) == (0.30, 0.3)


def test_select_s3_tie_goes_to_first_in_grid_order():
    spx, vix = _spike_market(spike_at=45)
    p = pd.Series(0.0, index=spx.index[:80])  # never hedged -> every pair ties
    assert select_s3(p, spx, vix, 1.0) == (0.10, 0.1)


def test_select_s3_empty_raises():
    spx, vix = _spike_market(spike_at=45)
    with pytest.raises(ValueError):
        select_s3(pd.Series(dtype=float), spx, vix, 1.0)


@pytest.mark.parametrize("std", [0.0, -1.0, np.nan, np.inf])
def test_s1_bad_std(std):
    with pytest.raises(ValueError):
        s1_weights(pd.Series([1.0]), std)
