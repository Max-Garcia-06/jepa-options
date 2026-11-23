import numpy as np
import pandas as pd
import pytest

from volj.config import RIDGE_ALPHAS
from volj.probes import brier, fit_probe, predict, select_param, sq_error


def _data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(rng.normal(size=(n, d)), index=idx)


def test_select_ridge_picks_small_alpha_on_clean_signal():
    X = _data(400)
    y = pd.Series(X.to_numpy() @ np.array([1.0, -2.0, 0.5, 3.0]), index=X.index)
    a = select_param("ridge", X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:])
    assert a == RIDGE_ALPHAS[0]


def test_logit_too_few_positives():
    X = _data()
    y = pd.Series(0.0, index=X.index)
    y.iloc[:9] = 1.0
    with pytest.raises(ValueError, match="too few"):
        fit_probe("logit", X, y, 1.0)


def test_logit_fit_predict_no_warning():
    X = _data(400)
    y = pd.Series((X[0] + 0.5 * np.random.default_rng(1).normal(size=400) > 1).astype(float), index=X.index)
    est = fit_probe("logit", X, y, 1.0)
    p = predict("logit", est, X)
    assert p.index.equals(X.index) and ((p >= 0) & (p <= 1)).all()
    c = select_param("logit", X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:])
    assert c > 0


def test_nan_y_dropped_nan_x_raises():
    X = _data()
    y = pd.Series(X[0].to_numpy(), index=X.index)
    y.iloc[:5] = np.nan
    fit_probe("ridge", X, y, 1.0)
    Xn = X.copy()
    Xn.iloc[50, 0] = np.nan
    with pytest.raises(ValueError):
        fit_probe("ridge", Xn, y, 1.0)


def test_brier():
    assert brier(pd.Series([0.2]), pd.Series([1.0])).iloc[0] == pytest.approx(0.64)


def test_sq_error():
    assert sq_error(pd.Series([1.0]), pd.Series([3.0])).iloc[0] == 4.0
