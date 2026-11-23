import numpy as np
import pandas as pd
import pytest
import torch

from volj.controls import fit_ols, har_regressors, har_vrp, lastday_control, pca_control
from volj.embed import embed
from volj.model import ContextEncoder


def _panel(n=200, f=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    return rng.standard_normal((n, f)), dates


def test_embed_uses_past_only():
    X, dates = _panel()
    torch.manual_seed(0)
    enc = ContextEncoder(X.shape[1]).eval()
    at = dates[[80, 120]]
    base = embed(enc, X, dates, at)
    X2 = X.copy()
    X2[121:] = np.random.default_rng(1).standard_normal(X2[121:].shape)
    assert base.shape == (2, 32) and list(base.columns[:2]) == ["z0", "z1"]
    assert base.dtypes.eq(np.float64).all()
    pd.testing.assert_frame_equal(base, embed(enc, X2, dates, at))


def test_embed_rejects_short_context():
    X, dates = _panel()
    enc = ContextEncoder(X.shape[1]).eval()
    with pytest.raises(ValueError):
        embed(enc, X, dates, dates[[10]])


def test_pca_fit_on_only():
    X, dates = _panel()
    fit_on = dates[70:130]
    at = dates[[90, 125]]
    a = pca_control(X, dates, at, fit_on, L=60, d=8)
    X2 = X.copy()
    X2[130:] = np.random.default_rng(2).standard_normal(X2[130:].shape)
    b = pca_control(X2, dates, at, fit_on, L=60, d=8)
    assert list(a.columns) == [f"p{i}" for i in range(8)]
    pd.testing.assert_frame_equal(a, b)


def test_lastday_control():
    X, dates = _panel()
    panel = pd.DataFrame(X, index=dates, columns=list("abcde"))
    out = lastday_control(panel, dates[[5, 9]])
    assert out.loc[dates[9], "c"] == X[9, 2]


def test_har_recovers_linear():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2000-01-01", periods=3000)
    r = rng.standard_normal(len(dates)) * 0.01
    r[::50] *= 4  # vary realized variance across regimes
    spx = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
    regs = har_regressors(spx)
    y = 0.5 * regs["rv2_22"] + 0.001 * rng.standard_normal(len(dates))
    _, beta = fit_ols(regs, y, dates)
    assert beta[2] == pytest.approx(0.5, abs=0.05)
    assert beta[0] == pytest.approx(0.0, abs=0.05)


def test_har_vrp_shape_and_alignment():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=400)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01)), index=dates)
    vix = pd.Series(20 + rng.standard_normal(400), index=dates)
    out = har_vrp(spx, vix, dates[30:250], dates[300:350], 1.0)
    assert out.index.equals(dates[300:350]) and out.notna().all()


def _har_inputs(n=500, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    r = rng.standard_normal(n) * 0.01
    r[::40] *= 4
    spx = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
    vix = pd.Series(20 + rng.standard_normal(n), index=dates)
    return spx, vix, dates


def test_har_vrp_rejects_label_overlap():
    spx, vix, dates = _har_inputs()
    at = dates[300:320]
    with pytest.raises(ValueError, match="overlap"):
        har_vrp(spx, vix, dates[30:295], at, 1.0)  # ends 5 days before at[0]
    # embargoed: last fit pos + H + 1 < pos(at[0])
    har_vrp(spx, vix, dates[30 : 300 - 23], at, 1.0)


def test_har_vrp_rejects_bad_at():
    spx, vix, dates = _har_inputs()
    with pytest.raises(ValueError):
        har_vrp(spx, vix, dates[30:200], dates[[5, 300]], 1.0)  # NaN regressors
    with pytest.raises(ValueError):
        har_vrp(spx, vix, dates[30:200], pd.DatetimeIndex(["1999-01-01"]), 1.0)


def test_har_vrp_end_to_end(monkeypatch):
    import volj.controls as controls
    from volj.targets import entry_k2

    spx, vix, dates = _har_inputs()
    regs = har_regressors(spx)
    y = 0.002 + 0.3 * regs["rv2_1"] + 0.2 * regs["rv2_5"] + 0.5 * regs["rv2_22"]
    monkeypatch.setattr(controls, "forward_rv2", lambda s: y)
    at = dates[300:330]
    out = har_vrp(spx, vix, dates[30:250], at, 1.0)
    forecast = y.loc[at]  # exact linear target -> OLS recovers it exactly
    expected = entry_k2(vix, 1.0).loc[at] - forecast
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-10)
