import json

import numpy as np
import pandas as pd
import pytest

import volj.pipeline as pl
from volj.pipeline import (apply_core_intersection, hash_inputs, input_keys, prelock_checks,
                           truncate_raw, verify_inputs)


@pytest.fixture(scope="module")
def raw():
    return pl.synthetic_raw()


@pytest.fixture(scope="module")
def ctx(raw):
    return pl.prepare(raw, "core", 1.0, (0, 1), 1, 1)


def test_input_keys_real_run_pins_raw_spx_and_encoders():
    keys = input_keys((0, 1), smoke=False)
    assert "raw/SPX.parquet" in keys and "raw/VIX.csv" in keys and "encoder_seed1.pt" in keys
    assert input_keys((0,), smoke=True) == ["encoder_seed0.pt"]


def test_raw_hash_mismatch_or_missing_spx_refused(tmp_path):
    data, out = tmp_path / "raw", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    keys = ["raw/VIX.csv", "raw/SPX.parquet", "encoder_seed0.pt"]
    (data / "VIX.csv").write_text("a")
    (data / "SPX.parquet").write_bytes(b"spx")
    (out / "encoder_seed0.pt").write_bytes(b"enc")
    rec = hash_inputs(keys, data, out)
    verify_inputs(rec, data, out)  # unchanged: fine
    (data / "VIX.csv").write_text("changed")
    with pytest.raises(RuntimeError, match=r"raw/VIX\.csv"):
        verify_inputs(rec, data, out)
    (data / "VIX.csv").write_text("a")
    (data / "SPX.parquet").unlink()  # missing SPX must refuse, never fetch
    with pytest.raises(RuntimeError, match=r"raw/SPX\.parquet"):
        verify_inputs(rec, data, out)
    with pytest.raises(RuntimeError, match="no input_sha256"):
        verify_inputs(None, data, out)


def test_truncate_raw_cuts_every_series(raw):
    cut = truncate_raw(raw, "2025-06-30")
    assert all(s.index[-1] <= pd.Timestamp("2025-06-30") for s in cut.values())
    assert cut["vix"].index[-1] == pd.Timestamp("2025-06-30")


def test_prelock_checks(ctx):
    end = ctx.dates[-1]
    from volj.config import STRESS_TEST
    prelock_checks(ctx, end, STRESS_TEST)
    with pytest.raises(RuntimeError, match="data_end"):
        prelock_checks(ctx, end - pd.Timedelta(days=30), STRESS_TEST)
    with pytest.raises(RuntimeError, match="stress window"):
        prelock_checks(ctx, end, [("2022-01-03", "2030-01-01")])
    with pytest.raises(RuntimeError, match="stress window"):
        prelock_checks(ctx, end, [("2019-01-03", "2019-02-01")])
    import dataclasses
    bad = dataclasses.replace(ctx, X=np.where(np.arange(ctx.X.size).reshape(ctx.X.shape) == 5,
                                              np.inf, ctx.X))
    with pytest.raises(RuntimeError, match="non-finite"):
        prelock_checks(bad, end, STRESS_TEST)
    empty = dataclasses.replace(ctx, splits={**ctx.splits, "test": ctx.splits["test"][:0]})
    with pytest.raises(RuntimeError, match="test split is empty"):
        prelock_checks(empty, end, STRESS_TEST)
    vix = ctx.vix.copy()
    vix.iloc[-3] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        prelock_checks(dataclasses.replace(ctx, vix=vix), end, STRESS_TEST)


def _res(**el):
    return {s: {"eligible_for_test": v} for s, v in el.items()}


def test_extended_cannot_rescue_core(tmp_path):
    core = tmp_path / "val_gate.json"
    core.write_text(json.dumps(_res(S1=True, S2=False, S3=False)))
    res = _res(S1=True, S2=True, S3=False)
    apply_core_intersection(res, ("S1", "S2", "S3"), core)
    assert [res[s]["eligible_for_test"] for s in ("S1", "S2", "S3")] == [True, False, False]
    assert res["S2"]["eligible_extended_only"] is True
    assert res["S2"]["eligibility_note"] == "core not eligible"


def test_extended_with_missing_core_gate_is_ineligible(tmp_path):
    res = _res(S1=True, S2=True, S3=True)
    apply_core_intersection(res, ("S1", "S2", "S3"), tmp_path / "nope.json")
    assert not any(res[s]["eligible_for_test"] for s in res)
    assert "missing" in res["S1"]["eligibility_note"]


def test_lookahead_check_runs_on_about_20_dates(raw, monkeypatch, caplog):
    seen = {}
    real = pl.assert_no_lookahead

    def spy(builder, r, dates):
        seen["dates"] = list(dates)
        return real(builder, r, dates)

    monkeypatch.setattr(pl, "assert_no_lookahead", spy)
    with caplog.at_level("INFO"):
        pl.prepare(raw, "core", 1.0, (0,), 1, 1, check_lookahead=True)
    assert 18 <= len(seen["dates"]) <= 20 and seen["dates"] == sorted(seen["dates"])
    assert "assert_no_lookahead" in caplog.text
    seen.clear()
    pl.prepare(raw, "core", 1.0, (0,), 1, 1)
    assert not seen


def test_restrict_raises_on_nan_weights(ctx):
    at = ctx.splits["val"]
    w = pd.Series(1.0, index=at)
    w.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN in weights"):
        pl._restrict(ctx, w, 1.0, *ctx.bounds("val"))


def test_pipeline_detects_oracle_leak(ctx):
    """Positive control through the real path: a column equal to the target must win G1 hugely."""
    import dataclasses

    from volj.probes import sq_error
    from volj.train import pretrain

    scaled = ctx.scaled.assign(oracle=ctx.vrp.fillna(0.0))  # illegal: the label itself
    octx = dataclasses.replace(ctx, scaled=scaled)  # X (JEPA/PCA input) stays legitimate
    encs = [pretrain(ctx.X, pl.train_positions(ctx), seed=k, epochs=1).eval() for k in ctx.seeds]
    params = pl.tune(octx, encs)
    val = octx.splits["val"]
    preds = pl.fit_predict(octx, encs, params, octx.splits["train"], val)
    y = octx.vrp.loc[val]
    r2 = 1 - sq_error(preds["vrp"]["lastday"], y).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.9
    r = pl.g1_for(octx, "S1", preds, val)
    assert r["pass"] is False
    assert r["per_control"]["lastday"]["mean_diff"] < 0  # control loss << JEPA loss
