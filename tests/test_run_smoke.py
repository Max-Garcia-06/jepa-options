import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "run_round1.py")


def run(stage, out, check=True):
    p = subprocess.run([sys.executable, SCRIPT, "--stage", stage, "--smoke", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    if check:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("r01")
    run("pretrain", d)
    run("val", d)
    return d


def test_smoke_val_runs(out):
    res = json.loads((out / "val_gate.json").read_text())
    for s in ("S1", "S2", "S3"):
        assert {"g1", "g2", "g3", "eligible_for_test"} <= set(res[s])
    assert res["S3"]["g1"]["shared_with"] == "S2"
    assert res["S3"]["g1"]["pass"] == res["S2"]["g1"]["pass"]
    assert set(res["S1"]["g2"]["comparators"]) == {
        "always_short", "term_structure", "har", "pca", "lastday"}
    assert "har" not in res["S2"]["g2"]["comparators"]
    assert set(res["S3"]["g2"]["comparators"]) == {
        "always_hedged", "never_hedged", "pca", "lastday"}
    assert set(res["cost_sensitivity"]["strategies"]) == {"S1", "S2", "S3"}
    assert len(res["meta"]["git_rev"]) >= 7 and len(res["meta"]["config_sha256"]) == 64
    assert (out / "val_gate.md").exists() and (out / "params.json").exists()
    assert (out / "encoder_seed0.pt").exists() and (out / "loss_history.json").exists()


def _forced_copy(out, tmp_path, name="copy"):
    """Own copy of the val outputs with S1 forced eligible (test-only edit)."""
    d = tmp_path / name
    shutil.copytree(out, d)
    gate = json.loads((d / "val_gate.json").read_text())
    gate["S1"]["eligible_for_test"] = True
    gate["S2"]["eligible_for_test"] = False
    gate["S3"]["eligible_for_test"] = False
    (d / "val_gate.json").write_text(json.dumps(gate))
    return d


def test_test_stage_locked_and_refuses_rerun(out, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    p = run("test", empty, check=False)
    assert p.returncode != 0 and "val gate not passed" in p.stderr
    assert not (empty / "test_look.lock").exists()

    d = _forced_copy(out, tmp_path)
    run("test", d)
    res = json.loads((d / "test_gate.json").read_text())
    assert set(res) == {"S1", "meta"}
    assert (d / "test_look.lock").exists()
    again = run("test", d, check=False)
    assert again.returncode != 0 and "already evaluated" in again.stderr


def test_preflight_refuses_mismatched_settings_before_lock(out, tmp_path):
    d = _forced_copy(out, tmp_path, "mismatch")
    p = subprocess.run([sys.executable, SCRIPT, "--stage", "test", "--smoke", "--cost", "2.0",
                        "--out", str(d)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "cost" in p.stderr
    assert not (d / "test_look.lock").exists()

    d2 = _forced_copy(out, tmp_path, "noenc")
    (d2 / "encoder_seed1.pt").unlink()
    p = run("test", d2, check=False)
    assert p.returncode != 0 and "encoder_seed1.pt" in p.stderr
    assert not (d2 / "test_look.lock").exists()


def test_smoke_refuses_repo_outputs_dir():
    p = run("val", ROOT / "outputs" / "r01" / "core", check=False)
    assert p.returncode != 0 and "smoke" in p.stderr


def test_val_extras(out):
    res = json.loads((out / "val_gate.json").read_text())
    assert isinstance(res["crash_threshold_train"], float)
    assert "h_notional" in res["S3"]["g2"]["comparators"]["always_hedged"]
    assert isinstance(res["meta"]["dirty"], bool)
    assert (out / "val_preds.parquet").exists()


@pytest.fixture(scope="module")
def ctx():
    from volj.pipeline import prepare, synthetic_raw
    return prepare(synthetic_raw(), "core", 1.0, (0, 1), 1, 1)


def test_pnl_window_bounds_and_zero_weights_off_split(ctx):
    import pandas as pd
    from volj.pipeline import _restrict
    assert ctx.bounds("val") == (pd.Timestamp("2017-01-03"), pd.Timestamp("2020-12-31"))
    lo, hi = ctx.bounds("test")
    assert lo == pd.Timestamp("2021-01-04") and hi == ctx.dates[-1]
    at = ctx.splits["test"]
    w = pd.Series(1.0, index=at)
    pnl = _restrict(ctx, w, 1.0, lo, hi)
    assert pnl.index[0] == lo and pnl.index[-1] == ctx.dates[-1]
    assert pnl.index[-1] > at[-1]  # last positions' P&L is not truncated at the last signal date
    assert (pnl.loc[:at[0]] == 0).all()  # nothing accrues before the first eval date
    wv = pd.Series(1.0, index=ctx.splits["val"])
    full = _restrict(ctx, wv, 1.0, ctx.dates[0], ctx.dates[-1])
    assert (full.loc[:ctx.splits["val"][0]] == 0).all()
    assert (full.loc[ctx.splits["val"][-1] + pd.Timedelta(days=60):] == 0).all()
