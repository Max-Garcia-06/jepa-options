import numpy as np
import pandas as pd
import pytest

from volj.config import FEATURE_SETS
from volj.data.features import (
    apply_scaler,
    assert_no_lookahead,
    build_panel,
    fit_scaler,
)


def make_raw(n=300, spx_returns=None):
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, n) if spx_returns is None else spx_returns
    raw = {"spx": pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="spx")}
    levels = {"vix": 20, "vix6m": 22, "vvix": 90, "skew": 130, "vix3m": 21, "vix9d": 19}
    for k, v in levels.items():
        raw[k] = pd.Series(v + rng.normal(0, 0.5, n), index=idx, name=k)
    return raw


def test_columns_and_order():
    raw = make_raw()
    assert list(build_panel(raw, "core").columns) == FEATURE_SETS["core"]
    assert list(build_panel(raw, "extended").columns) == FEATURE_SETS["extended"]


def test_rv21_value():
    a = 0.20 / np.sqrt(252)
    r = np.where(np.arange(300) % 2 == 0, a, -a)
    panel = build_panel(make_raw(spx_returns=r), "core")
    assert len(panel) > 200
    np.testing.assert_allclose(panel["rv21"].to_numpy(), 0.20, atol=1e-9)


def test_ffill_limit():
    raw = make_raw()
    full = len(build_panel(raw, "core"))
    gap = raw["spx"].index[150:153]
    raw["vvix"] = raw["vvix"].drop(gap)
    panel = build_panel(raw, "core")
    assert len(panel) == full - 1
    assert gap[0] in panel.index and gap[1] in panel.index
    assert gap[2] not in panel.index


def test_no_lookahead_passes():
    raw = make_raw()
    assert_no_lookahead(lambda r: build_panel(r, "core"), raw, raw["spx"].index[100:110])


def test_no_lookahead_catches_leak():
    def leaky(r):
        p = build_panel(r, "core")
        p["leak"] = p["ret"].shift(-1)
        return p

    raw = make_raw()
    with pytest.raises(AssertionError, match="leak"):
        assert_no_lookahead(leaky, raw, raw["spx"].index[100:110])


def test_scaler_uses_fit_dates_only():
    panel = build_panel(make_raw(), "core")
    mean, std = fit_scaler(panel, panel.index[:100])
    pd.testing.assert_series_equal(mean, panel.iloc[:100].mean())
    pd.testing.assert_series_equal(std, panel.iloc[:100].std())
    scaled = apply_scaler(panel, mean, std)
    np.testing.assert_allclose(scaled.iloc[:100].mean().to_numpy(), 0, atol=1e-10)
