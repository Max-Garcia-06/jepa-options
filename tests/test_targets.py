import numpy as np
import pandas as pd
import pytest

from volj.targets import (crash_label, crash_threshold, entry_k2, forward_rv2,
                          vrp_target)


def _spx_from_returns(r):
    idx = pd.bdate_range("2020-01-01", periods=len(r) + 1)
    return pd.Series(np.exp(np.concatenate([[0.0], np.cumsum(r)])), index=idx)


def test_forward_rv2_alignment():
    a = 0.20 / np.sqrt(252)
    r = np.tile([a, -a], 40)
    spx = _spx_from_returns(r)
    rv = forward_rv2(spx)
    assert rv.iloc[:-22].notna().all()
    assert np.allclose(rv.iloc[:-22], 0.04, atol=1e-12)
    assert rv.iloc[-22:].isna().all()


def test_forward_rv2_uses_t_plus_2():
    r = np.zeros(60)
    r[29] = 0.05  # log return at position 30 of the price series
    spx = _spx_from_returns(r)
    rv = forward_rv2(spx)
    expected = 252 / 21 * 0.05**2
    assert rv.iloc[8] == pytest.approx(expected, abs=1e-12)
    assert rv.iloc[28] == pytest.approx(expected, abs=1e-12)
    assert rv.iloc[7] == pytest.approx(0.0, abs=1e-12)
    assert rv.iloc[29] == pytest.approx(0.0, abs=1e-12)


def test_vrp_cost():
    a = 0.20 / np.sqrt(252)
    spx = _spx_from_returns(np.tile([a, -a], 40))
    vix = pd.Series(20.0, index=spx.index)
    v = vrp_target(spx, vix, cost_volpt=1.0)
    assert v.iloc[0] == pytest.approx(0.19**2 - 0.04, abs=1e-12)
    assert entry_k2(vix, 1.0).iloc[-1] != entry_k2(vix, 1.0).iloc[-1]  # NaN tail
    assert v.iloc[-22:].isna().all()


def test_crash_label_nan_preserved():
    vrp = pd.Series([np.nan, -1.0, 0.5, 0.0])
    lab = crash_label(vrp, 0.0)
    assert np.isnan(lab.iloc[0])
    assert lab.tolist()[1:] == [1.0, 0.0, 1.0]


def test_crash_threshold_fit_dates_only():
    vrp = pd.Series(np.arange(100.0), index=pd.bdate_range("2020-01-01", periods=100))
    thr = crash_threshold(vrp, vrp.index[:50])
    assert thr == pytest.approx(np.quantile(np.arange(50.0), 0.10))
