import json

import numpy as np
import pandas as pd
import pytest

from volj.gate import acquire_test_lock, block_bootstrap_ci, g1, g2, g3, write_report


def test_ci_excludes_zero_for_clear_signal():
    d = 1 + 0.1 * np.random.default_rng(0).standard_normal(1000)
    lo, hi = block_bootstrap_ci(d)
    assert lo > 0


def test_ci_includes_zero_for_noise():
    d = np.random.default_rng(0).standard_normal(1000)
    lo, hi = block_bootstrap_ci(d)
    assert lo < 0 < hi


def _idx(n=500):
    return pd.bdate_range("2017-01-03", periods=n)


def _series(mean, n=500, seed=0):
    return pd.Series(mean + 0.1 * np.random.default_rng(seed).standard_normal(n), index=_idx(n))


def test_g1_requires_all_controls():
    jepa = _series(1.0, seed=1)
    seeds = [_series(1.0, seed=10 + i) for i in range(5)]
    controls = {"pca": _series(1.5, seed=2), "last": _series(0.8, seed=3)}
    r = g1(jepa, seeds, controls)
    assert r["pass"] is False
    assert r["per_control"]["pca"]["ci"][0] > 0


def test_g1_seed_rule():
    jepa = _series(1.0, seed=1)
    seeds = [_series(1.0, seed=10 + i) for i in range(3)] + [_series(2.0, seed=20 + i) for i in range(2)]
    controls = {"pca": _series(1.5, seed=2), "last": _series(1.5, seed=3)}
    r = g1(jepa, seeds, controls)
    assert r["per_control"]["pca"]["seeds_positive"] == 3
    assert r["pass"] is False
    assert g1(jepa, seeds, controls, min_seeds=3)["pass"] is True


def test_g1_empty_intersection_and_nan():
    a = _series(1.0, 50)
    b = pd.Series(1.0, index=pd.bdate_range("2030-01-01", periods=50))
    with pytest.raises(ValueError):
        g1(a, [a], {"c": b})
    c = a.copy()
    c.iloc[3] = np.nan
    with pytest.raises(ValueError):
        g1(a, [a], {"c": c})


def _pnl(mean, sd=1.0, n=400, seed=0):
    return pd.Series(mean + sd * np.random.default_rng(seed).standard_normal(n), index=_idx(n))


def test_g2_jepa_vs_strategy_pass():
    pnl = _pnl(0.3, seed=1)
    comps = {"always_short": _pnl(0.0, seed=2), "pca": _pnl(0.6, seed=3)}
    bench = _pnl(0.0, sd=3.0, seed=4)
    r = g2(pnl, comps, bench, {"always_short"})
    assert r["strategy_pass"] is True and r["jepa_pass"] is False
    with pytest.raises(ValueError):
        g2(pnl, comps, bench, {"nope"})


def test_g2_tie_is_not_beating_and_dd_required():
    pnl = _pnl(0.3, seed=1)
    r = g2(pnl, {"same": pnl.copy()}, pnl, {"same"})
    assert r["strategy_pass"] is False
    r = g2(pnl, {"w": _pnl(0.0, seed=2)}, pnl * 0.1, {"w"})
    assert r["strategy_pass"] is False


def test_g3_window():
    idx = _idx(100)
    bench = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)
    w1 = (str(idx[10].date()), str(idx[20].date()))
    w2 = (str(idx[50].date()), str(idx[60].date()))
    bench.loc[idx[55]] = -1.0
    pnl.loc[idx[55]] = -0.5
    assert g3(pnl, bench, [w1, w2])["pass"] is True
    pnl.loc[idx[15]] = -1.0
    assert g3(pnl, bench, [w1, w2])["pass"] is False


def _gate(tmp_path, eligible):
    (tmp_path / "val_gate.json").write_text(json.dumps(
        {"git_rev": "x", "S1": {"eligible_for_test": eligible}, "S2": {"eligible_for_test": False}}))


def test_lock_refuses_second_look(tmp_path):
    _gate(tmp_path, True)
    acquire_test_lock(tmp_path)
    lock = json.loads((tmp_path / "test_look.lock").read_text())
    assert set(lock) == {"timestamp", "git_rev"}
    before = (tmp_path / "test_look.lock").read_text()
    with pytest.raises(RuntimeError, match="test set already evaluated"):
        acquire_test_lock(tmp_path)
    assert (tmp_path / "test_look.lock").read_text() == before


def test_lock_requires_val_pass(tmp_path):
    with pytest.raises(RuntimeError, match="val gate not passed"):
        acquire_test_lock(tmp_path)
    _gate(tmp_path, False)
    with pytest.raises(RuntimeError, match="val gate not passed"):
        acquire_test_lock(tmp_path)
    assert not (tmp_path / "test_look.lock").exists()


def test_write_report_handles_numpy(tmp_path):
    res = {"S1": {"pass": np.bool_(True), "sharpe": np.float64(1.5),
                  "g1": {"per_control": {"pca": {"ci": (np.float64(0.1), 0.2)}}}}}
    write_report(res, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["S1"]["sharpe"] == 1.5
    assert "S1" in (tmp_path / "r.md").read_text()
