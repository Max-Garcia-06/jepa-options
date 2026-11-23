"""Round-1 staged pipeline: fetch -> pretrain -> val -> test (test runs once, locked).

uv run scripts/run_round1.py --stage {fetch,pretrain,val,test} [--feature-set core|extended]
                             [--cost 1.0] [--smoke] [--out DIR]
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import torch

from volj import config
from volj.config import COST_SENS, COST_VOLPT, STRESS_TEST, STRESS_VAL
from volj.data.fetch import load_raw
from volj.gate import REPO_ROOT, acquire_test_lock, write_report
from volj.model import ContextEncoder
from volj.pipeline import (STRATEGIES, Ctx, fit_predict, gates, meta, prepare,
                           synthetic_raw, train_positions, tune)
from volj.train import pretrain

log = logging.getLogger("run_round1")
CBOE_SYMS = ["VIX", "VIX6M", "VVIX", "SKEW", "VIX3M", "VIX9D"]
SMOKE = {"seeds": (0, 1), "epochs": 1, "min_seeds": 1}


def make_ctx(args, out: Path) -> Ctx:
    if args.smoke:
        raw, kw = synthetic_raw(), SMOKE
    else:
        raw = load_raw(REPO_ROOT / "data" / "raw", CBOE_SYMS)
        kw = {"seeds": config.SEEDS, "epochs": config.EPOCHS, "min_seeds": config.MIN_SEEDS_POSITIVE}
    return prepare(raw, args.feature_set, args.cost, **kw)


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


def _info(args, ctx: Ctx) -> dict:
    return meta({"feature_set": args.feature_set, "cost": args.cost, "smoke": args.smoke,
                 "seeds": list(ctx.seeds), "n_dates": {k: len(v) for k, v in ctx.splits.items()}})


def stage_val(args, out: Path) -> None:
    ctx = make_ctx(args, out)
    if args.smoke and not (out / f"encoder_seed{ctx.seeds[0]}.pt").exists():
        stage_pretrain(args, out, ctx)  # smoke convenience only; real runs must pretrain first
    encs = load_encoders(ctx, out)
    params = tune(ctx, encs)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    train, val = ctx.splits["train"], ctx.splits["val"]
    preds = fit_predict(ctx, encs, params, train, val)
    res = gates(ctx, STRATEGIES, preds, params, val, "val", STRESS_VAL, ctx.cost, train)
    sens = gates(ctx, STRATEGIES, preds, params, val, "val", STRESS_VAL, COST_SENS, train,
                 with_g1=False)
    res["cost_sensitivity"] = {"cost": COST_SENS, "report_only": True, "strategies": sens}
    res["params"] = params
    res["meta"] = _info(args, ctx)
    write_report(res, out / "val_gate.json")
    print({s: res[s]["eligible_for_test"] for s in STRATEGIES})


def stage_test(args, out: Path) -> None:
    acquire_test_lock(out)  # first: refuses a re-run or an unpassed val gate
    val_gate = json.loads((out / "val_gate.json").read_text())
    eligible = [s for s in STRATEGIES if val_gate[s]["eligible_for_test"]]
    params = json.loads((out / "params.json").read_text())
    ctx = make_ctx(args, out)
    encs = load_encoders(ctx, out)
    fit = ctx.splits["train"].append(ctx.splits["val"])
    at = ctx.splits["test"]
    preds = fit_predict(ctx, encs, params, fit, at)
    res = gates(ctx, eligible, preds, params, at, "test", STRESS_TEST, ctx.cost, fit)
    res["meta"] = _info(args, ctx)
    write_report(res, out / "test_gate.json")
    print({s: res[s]["eligible_for_test"] for s in eligible})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "pretrain", "val", "test"])
    ap.add_argument("--feature-set", default="core", choices=sorted(config.FEATURE_SETS))
    ap.add_argument("--cost", type=float, default=COST_VOLPT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out or (Path(tempfile.mkdtemp(prefix="r01_smoke_")) if args.smoke
                       else REPO_ROOT / "outputs" / "r01" / args.feature_set)
    out.mkdir(parents=True, exist_ok=True)
    {"fetch": stage_fetch, "pretrain": stage_pretrain, "val": stage_val,
     "test": stage_test}[args.stage](args, out)


if __name__ == "__main__":
    main()
