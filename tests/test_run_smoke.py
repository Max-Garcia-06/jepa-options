import json
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


def test_test_stage_locked_and_refuses_rerun(out, tmp_path):
    # val gate not passed -> refuse, no lock created
    empty = tmp_path / "none"
    empty.mkdir()
    p = run("test", empty, check=False)
    assert p.returncode != 0 and "val gate not passed" in p.stderr
    assert not (empty / "test_look.lock").exists()

    # force one eligible strategy (test-only edit) to exercise the test path
    gate = json.loads((out / "val_gate.json").read_text())
    gate["S1"]["eligible_for_test"] = True
    gate["S2"]["eligible_for_test"] = False
    gate["S3"]["eligible_for_test"] = False
    (out / "val_gate.json").write_text(json.dumps(gate))
    run("test", out)
    res = json.loads((out / "test_gate.json").read_text())
    assert set(res) == {"S1", "meta"}
    assert (out / "test_look.lock").exists()
    again = run("test", out, check=False)
    assert again.returncode != 0 and "already evaluated" in again.stderr
