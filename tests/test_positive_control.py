import numpy as np
import pandas as pd
import pytest

from volj.config import COST_VOLPT
from volj.data.features import assert_no_lookahead, build_panel
from volj.probes import fit_probe, predict
from volj.targets import vrp_target


def make_raw(n=1500):
    idx = pd.bdate_range("2010-01-01", periods=n, name="date")
    rng = np.random.default_rng(0)
    regime = (np.arange(n) // 120) % 2
    vol = np.where(regime == 0, 0.007, 0.018)
    r = rng.normal(0, 1, n) * vol
    spx = pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="spx")
    rv = pd.Series(r, index=idx).rolling(21, min_periods=1).std().bfill() * np.sqrt(252) * 100
    raw = {"spx": spx, "vix": (rv + 3 + rng.normal(0, 0.5, n)).rename("vix")}
    for k, v in {"vix6m": 22, "vvix": 90, "skew": 130, "vix3m": 21, "vix9d": 19}.items():
        raw[k] = pd.Series(v + rng.normal(0, 0.5, n), index=idx, name=k)
    return raw


def _with_oracle(raw):
    p = build_panel(raw, "core")
    p["oracle"] = vrp_target(raw["spx"], raw["vix"], COST_VOLPT).reindex(p.index)
    return p


def test_positive_control():
    raw = make_raw()
    panel = _with_oracle(raw)
    y = vrp_target(raw["spx"], raw["vix"], COST_VOLPT).reindex(panel.index)
    lab = y.dropna().index
    assert len(lab) >= 1300
    fit, ev = lab[:1000], lab[1000:1300]
    X = panel[["oracle"]]
    est = fit_probe("ridge", X.loc[fit], y.loc[fit], 1e-3)
    pred = predict("ridge", est, X.loc[ev])
    yt = y.loc[ev]
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    assert r2 > 0.9
    assert yt.std() > 0


def test_oracle_detected_by_lookahead_check():
    raw = make_raw()
    with pytest.raises(AssertionError, match="oracle"):
        assert_no_lookahead(_with_oracle, raw, raw["spx"].index[200:210])
