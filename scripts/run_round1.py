"""Round-1 staged pipeline: fetch -> pretrain -> val -> test (test runs once, locked).

uv run scripts/run_round1.py --stage {fetch,pretrain,val,test} [--feature-set core|extended]
                             [--cost 1.0] [--smoke] [--out DIR]
"""

import argparse
import json
import logging
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from volj import config
from volj.config import COST_SENS, COST_VOLPT, STRESS_TEST, STRESS_VAL
from volj.data.fetch import CBOE_SYMS, load_raw
from volj.gate import REPO_ROOT, acquire_test_lock, write_report
from volj.model import ContextEncoder
from volj.pipeline import (STRATEGIES, Ctx, apply_core_intersection, compute_pnls,
                           fit_predict, gates, hash_inputs, input_keys, meta, persist_artifacts,
                           prelock_checks, prepare, require_clean_tree, require_no_prior_val,
                           synthetic_raw, train_positions, truncate_raw, tune, verify_inputs)
from volj.splits import test_end
from volj.train import pretrain

log = logging.getLogger("run_round1")
SMOKE = {"seeds": (0, 1), "epochs": 1, "min_seeds": 1}


def run_settings(args) -> dict:
    if args.smoke:
        return SMOKE
    return {"seeds": config.SEEDS, "epochs": config.EPOCHS, "min_seeds": config.MIN_SEEDS_POSITIVE}


def data_dir(args) -> Path:
    return getattr(args, "data_dir", None) or REPO_ROOT / "data" / "raw"


def make_ctx(args, out: Path, data_end=None) -> Ctx:
    """Context for the run; with data_end, raw data is truncated there before anything is built."""
    raw = synthetic_raw() if args.smoke else load_raw(data_dir(args), CBOE_SYMS)
    if data_end is not None:
        raw = truncate_raw(raw, data_end)
    return prepare(raw, args.feature_set, args.cost, check_lookahead=not args.smoke,
                   **run_settings(args))


def stage_fetch(args, out: Path) -> None:
    raw = load_raw(REPO_ROOT / "data" / "raw", CBOE_SYMS, refresh=True)
    for k, s in raw.items():
        print(f"{k}: {len(s)} rows, {s.index[0].date()} -> {s.index[-1].date()}")


def stage_pretrain(args, out: Path, ctx: Ctx = None) -> None:
    ctx = ctx or make_ctx(args, out)
    out.mkdir(parents=True, exist_ok=True)
    pos, hist = train_positions(ctx), {}
    for k in ctx.seeds:
        enc = pretrain(ctx.X, pos, seed=k, epochs=ctx.epochs)
        torch.save(enc.state_dict(), out / f"encoder_seed{k}.pt")
        hist[str(k)] = [float(v) for v in enc.loss_history]
        log.info("seed %d pretrained", k)
    (out / "loss_history.json").write_text(json.dumps(hist))


def load_encoders(ctx: Ctx, out: Path) -> list:
    encs = []
    for k in ctx.seeds:
        p = out / f"encoder_seed{k}.pt"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run --stage pretrain first")
        enc = ContextEncoder(ctx.X.shape[1])
        enc.load_state_dict(torch.load(p, map_location="cpu"))
        encs.append(enc.eval())
    return encs


def _info(args, ctx: Ctx, out: Path) -> dict:
    keys = input_keys(ctx.seeds, args.smoke)
    return meta({"feature_set": args.feature_set, "cost": args.cost, "smoke": args.smoke,
                 "seeds": list(ctx.seeds), "n_dates": {k: len(v) for k, v in ctx.splits.items()},
                 "data_end": str(ctx.dates[-1].date()), "test_end": str(test_end(ctx.dates).date()),
                 "input_sha256": hash_inputs(keys, data_dir(args), out)})


def preflight(args, out: Path):
    """Checks that make no test-data-dependent decision; must pass before the lock is taken.

    Returns (ctx, encoders), or None when there is no val gate (the lock then refuses).
    """
    gate = out / "val_gate.json"
    if not gate.exists():
        return None
    m = json.loads(gate.read_text()).get("meta", {})
    for key, now in (("feature_set", args.feature_set), ("cost", args.cost), ("smoke", args.smoke)):
        if m.get(key) != now:
            raise RuntimeError(f"test settings differ from val run: {key} was {m.get(key)!r}, "
                               f"now {now!r}")
    try:
        params = json.loads((out / "params.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"params.json missing or unreadable: {e}") from e
    for k in run_settings(args)["seeds"]:
        if not (out / f"encoder_seed{k}.pt").exists():
            raise RuntimeError(f"missing encoder_seed{k}.pt; cannot run test stage")
    verify_inputs(m.get("input_sha256"), data_dir(args), out)  # no fetch happens before this
    ctx = make_ctx(args, out, data_end=m["data_end"])
    prelock_checks(ctx, m["data_end"], STRESS_TEST)
    encs = load_encoders(ctx, out)
    fit = ctx.splits["train"].append(ctx.splits["val"])
    fit_predict(ctx, encs, params, fit[:-5], fit[-5:])  # train+val probe fit check (no test data)
    return ctx, encs


def stage_val(args, out: Path) -> None:
    if not args.smoke:
        require_clean_tree()
        require_no_prior_val(out)
    ctx = make_ctx(args, out)
    if args.smoke and not (out / f"encoder_seed{ctx.seeds[0]}.pt").exists():
        stage_pretrain(args, out, ctx)  # smoke convenience only; real runs must pretrain first
    encs = load_encoders(ctx, out)
    params = tune(ctx, encs)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    train, val = ctx.splits["train"], ctx.splits["val"]
    preds = fit_predict(ctx, encs, params, train, val)
    sps = compute_pnls(ctx, STRATEGIES, preds, params, val, "val", ctx.cost, train)
    persist_artifacts(out, "val", preds, sps)  # before any gate is computed
    res = gates(ctx, STRATEGIES, preds, params, val, "val", STRESS_VAL, ctx.cost, train, sps=sps)
    sens = gates(ctx, STRATEGIES, preds, params, val, "val", STRESS_VAL, COST_SENS, train,
                 with_g1=False)
    if args.feature_set == "extended":
        core = getattr(args, "core_gate", None) or REPO_ROOT / "outputs" / "r01" / "core" / "val_gate.json"
        apply_core_intersection(res, STRATEGIES, core)
    res["crash_threshold_train"] = ctx.crash_thr
    res["cost_sensitivity"] = {"cost": COST_SENS, "report_only": True, "strategies": sens}
    res["params"] = params
    res["meta"] = _info(args, ctx, out)
    write_report(res, out / "val_gate.json")
    print({s: res[s]["eligible_for_test"] for s in STRATEGIES})


def _run_test(args, out: Path, ctx: Ctx, encs: list) -> None:
    val_gate = json.loads((out / "val_gate.json").read_text())
    eligible = [s for s in STRATEGIES if val_gate[s]["eligible_for_test"]]
    params = json.loads((out / "params.json").read_text())
    fit = ctx.splits["train"].append(ctx.splits["val"])
    at = ctx.splits["test"]
    preds = fit_predict(ctx, encs, params, fit, at)
    sps = compute_pnls(ctx, eligible, preds, params, at, "test", ctx.cost, fit)
    persist_artifacts(out, "test", preds, sps)  # before any gate is computed
    res = gates(ctx, eligible, preds, params, at, "test", STRESS_TEST, ctx.cost, fit, sps=sps)
    res["meta"] = _info(args, ctx, out)
    write_report(res, out / "test_gate.json")
    print({s: res[s]["eligible_for_test"] for s in eligible})


def stage_test(args, out: Path) -> None:
    if not args.smoke:
        require_clean_tree()
    pre = preflight(args, out)
    acquire_test_lock(out)  # refuses a re-run or an unpassed val gate
    try:
        _run_test(args, out, *pre)
    except Exception:
        (out / "test_failed.json").write_text(json.dumps(
            {"timestamp": datetime.now(timezone.utc).isoformat(),
             "traceback": traceback.format_exc()}, indent=2))
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "pretrain", "val", "test"])
    ap.add_argument("--feature-set", default="core", choices=sorted(config.FEATURE_SETS))
    ap.add_argument("--cost", type=float, default=COST_VOLPT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None, help="raw data cache (default data/raw)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    smoke_dir_ok = lambda p: not p.resolve().is_relative_to(REPO_ROOT / "outputs" / "r01")  # noqa
    out = args.out or (Path(tempfile.mkdtemp(prefix="r01_smoke_")) if args.smoke
                       else REPO_ROOT / "outputs" / "r01" / args.feature_set)
    if args.smoke and not smoke_dir_ok(out):
        ap.error("--smoke must not write under outputs/r01; use a separate --out")
    out.mkdir(parents=True, exist_ok=True)
    {"fetch": stage_fetch, "pretrain": stage_pretrain, "val": stage_val,
     "test": stage_test}[args.stage](args, out)


if __name__ == "__main__":
    main()
