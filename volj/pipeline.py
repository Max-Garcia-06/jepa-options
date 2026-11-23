"""Round-1 orchestration helpers: data prep, inner tuning, probe fits, gate evaluation.

Nothing here mutates volj.config; smoke-mode overrides arrive as parameters.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from volj.config import FEATURE_SETS, L, MAX_FFILL, SPLITS
from volj.controls import har_vrp, lastday_control, pca_control
from volj.data.features import apply_scaler, assert_no_lookahead, build_panel, fit_scaler
from volj.data.fetch import CBOE_SYMS
from volj.embed import embed
from volj.gate import REPO_ROOT, g1, g2, g3, git_dirty, git_rev
from volj.pnl import long_spx_pnl, short_var_ladder_pnl
from volj.probes import brier, fit_probe, predict, select_param, sq_error
from volj.splits import split_dates
from volj.strategies import (always_hedged, always_short, s1_weights, s2_weights,
                             s3_pnl, select_s3, term_structure)
from volj.targets import crash_label, crash_threshold, vrp_target

log = logging.getLogger(__name__)

STRATEGIES = ("S1", "S2", "S3")
INPUTS = ("pca", "lastday")  # non-JEPA probe inputs


def config_sha256() -> str:
    return hashlib.sha256((REPO_ROOT / "volj" / "config.py").read_bytes()).hexdigest()


def meta(extra: dict) -> dict:
    return {"git_rev": git_rev(), "dirty": git_dirty(), "config_sha256": config_sha256(), **extra}


@dataclass
class Ctx:
    panel: pd.DataFrame
    scaled: pd.DataFrame
    X: np.ndarray
    dates: pd.DatetimeIndex
    spx: pd.Series
    vix: pd.Series
    vrp: pd.Series
    splits: dict
    cost: float
    seeds: tuple
    epochs: int
    min_seeds: int
    crash_thr: float  # train-split threshold: the fixed label definition
    crash_y: pd.Series

    def bounds(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        lo, hi = SPLITS[name]
        return pd.Timestamp(lo), pd.Timestamp(hi) if hi else self.dates[-1]


def prepare(raw: dict, feature_set: str, cost: float, seeds: tuple, epochs: int,
            min_seeds: int, check_lookahead: bool = False) -> Ctx:
    """Panel, train-only scaler, VRP target, and context-filtered split dates."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    panel = build_panel(raw, feature_set)
    dates = panel.index
    if check_lookahead:
        t0 = time.perf_counter()
        pos = np.unique(np.linspace(0, len(dates) - 1, 20).round().astype(int))
        assert_no_lookahead(lambda r: build_panel(r, feature_set), raw, list(dates[pos]))
        log.info("assert_no_lookahead on %d panel dates: %.1fs", len(pos), time.perf_counter() - t0)
    mean, std = fit_scaler(panel, split_dates(dates, "train"))
    scaled = apply_scaler(panel, mean, std)
    spx = raw["spx"]
    vix = raw["vix"].reindex(spx.index).ffill(limit=MAX_FFILL)
    vrp = vrp_target(spx, vix, cost).reindex(dates)
    splits = {}
    for name in ("train", "val", "test", "inner_fit", "inner_select"):
        d = split_dates(dates, name)
        keep = d[dates.get_indexer(d) >= L - 1]
        log.info("split %s: dropped %d dates lacking %d-day context", name, len(d) - len(keep), L)
        splits[name] = keep
    thr = crash_threshold(vrp, splits["train"])
    return Ctx(panel, scaled, scaled.to_numpy(float), dates, spx, vix, vrp, splits, cost,
               tuple(seeds), epochs, min_seeds, thr, crash_label(vrp, thr))


def train_positions(ctx: Ctx) -> np.ndarray:
    return ctx.dates.get_indexer(ctx.splits["train"])


def _inputs(ctx: Ctx, encoders: list, fit: pd.DatetimeIndex, at: pd.DatetimeIndex) -> dict:
    """Probe feature frames on `at` for jepa{k}, pca (fit on `fit`), lastday."""
    out = {f"jepa{k}": embed(enc, ctx.X, ctx.dates, at) for k, enc in zip(ctx.seeds, encoders)}
    out["pca"] = pca_control(ctx.X, ctx.dates, at, fit_on=fit)
    out["lastday"] = lastday_control(ctx.scaled, at)
    return out


def tune(ctx: Ctx, encoders: list) -> dict:
    """Inner-split tuning only: probe params, S3 (tau, h). Never sees val/test dates."""
    fit, sel = ctx.splits["inner_fit"], ctx.splits["inner_select"]
    feats = _inputs(ctx, encoders, fit, fit.append(sel))
    thr = crash_threshold(ctx.vrp, fit)
    yc = crash_label(ctx.vrp, thr)
    probe, p_sel = {}, {}
    for name, F in feats.items():
        ridge = select_param("ridge", F.loc[fit], ctx.vrp.loc[fit], F.loc[sel], ctx.vrp.loc[sel])
        c = select_param("logit", F.loc[fit], yc.loc[fit], F.loc[sel], yc.loc[sel])
        probe[name] = {"ridge": ridge, "logit": c}
        est = fit_probe("logit", F.loc[fit], yc.loc[fit], c)
        p_sel[name] = predict("logit", est, F.loc[sel])
    p_sel["jepa"] = pd.concat([p_sel[f"jepa{k}"] for k in ctx.seeds], axis=1).mean(axis=1)
    s3 = {}
    for name in ("jepa", *INPUTS):
        tau, h = select_s3(p_sel[name], ctx.spx, ctx.vix, ctx.cost)
        s3[name] = {"tau": tau, "h": h}
    return {"inner_crash_threshold": thr, "probe": probe, "s3": s3}


def fit_predict(ctx: Ctx, encoders: list, params: dict, fit: pd.DatetimeIndex,
                at: pd.DatetimeIndex) -> dict:
    """Refit probes on `fit` with tuned params; predict on `at`. Seed average = mean."""
    yc = ctx.crash_y
    feats = _inputs(ctx, encoders, fit, fit.append(at))
    vrp_p, p_c = {}, {}
    for name, F in feats.items():
        pr = params["probe"][name]
        vrp_p[name] = predict("ridge", fit_probe("ridge", F.loc[fit], ctx.vrp.loc[fit], pr["ridge"]),
                              F.loc[at])
        p_c[name] = predict("logit", fit_probe("logit", F.loc[fit], yc.loc[fit], pr["logit"]),
                            F.loc[at])
    out = {"vrp": vrp_p, "p": p_c}
    for key, d in (("vrp", vrp_p), ("p", p_c)):
        seeds = [d.pop(f"jepa{k}") for k in ctx.seeds]
        d["jepa"] = pd.concat(seeds, axis=1).mean(axis=1)
        d["jepa_seeds"] = seeds
    return out


def _restrict(ctx: Ctx, w: pd.Series, cost: float, lo, hi) -> pd.Series:
    """Ladder P&L from weights nonzero only on w's (evaluation) dates, cut to [lo, hi]."""
    if w.isna().any():
        raise ValueError(f"NaN in weights on {int(w.isna().sum())} dates")
    return short_var_ladder_pnl(w.reindex(ctx.spx.index).fillna(0.0), ctx.spx, ctx.vix,
                                cost).loc[lo:hi]  # fillna: off-split dates carry zero weight


def strategy_pnls(ctx: Ctx, strat: str, preds: dict, params: dict, at: pd.DatetimeIndex,
                  split: str, cost: float, har_fit: pd.DatetimeIndex) -> dict:
    """{'pnl','comparators','dd_bench','strategy_only','weights'} for one strategy on `split`.

    'weights' maps 'strategy' and each comparator to its weights on `at` (hedge indicator
    for S3).
    """
    lo, hi = ctx.bounds(split)
    pnl = lambda w: _restrict(ctx, w, cost, lo, hi)  # noqa: E731
    if strat == "S3":
        h = params["s3"]["jepa"]["h"]

        def hedge_w(name):
            return (preds["p"][name] > params["s3"][name]["tau"]).astype(float)

        def s3(name):
            q = params["s3"][name]
            return s3_pnl(preds["p"][name], q["tau"], q["h"], ctx.spx, ctx.vix, cost).loc[lo:hi]
        comps = {"always_hedged": always_hedged(at, h, ctx.spx, ctx.vix, cost).loc[lo:hi],
                 "never_hedged": long_spx_pnl(ctx.spx).loc[lo:hi],
                 "pca": s3("pca"), "lastday": s3("lastday")}
        weights = {"strategy": hedge_w("jepa"), "pca": hedge_w("pca"),
                   "lastday": hedge_w("lastday"), "always_hedged": always_short(at),
                   "never_hedged": pd.Series(0.0, index=at)}
        return {"pnl": s3("jepa"), "comparators": comps, "dd_bench": comps["never_hedged"],
                "strategy_only": {"always_hedged", "never_hedged"}, "hedge_h": h,
                "weights": weights}
    term = term_structure(ctx.panel["vix"].loc[at], ctx.panel["vix6m"].loc[at])
    comps = {"always_short": pnl(always_short(at)), "term_structure": pnl(term)}
    cw = {"always_short": always_short(at), "term_structure": term}
    if strat == "S1":
        tstd = float(ctx.vrp.loc[ctx.splits["train"]].std())
        weights = {n: s1_weights(preds["vrp"][n], tstd) for n in ("jepa", *INPUTS)}
        cw["har"] = s1_weights(har_vrp(ctx.spx, ctx.vix, har_fit, at, ctx.cost), tstd)
        comps["har"] = pnl(cw["har"])
        only = {"always_short", "term_structure", "har"}
    else:
        weights = {n: s2_weights(preds["p"][n]) for n in ("jepa", *INPUTS)}
        only = {"always_short", "term_structure"}
    for n in INPUTS:
        comps[n] = pnl(weights[n])
    return {"pnl": pnl(weights["jepa"]), "comparators": comps, "dd_bench": comps["always_short"],
            "strategy_only": only,
            "weights": {"strategy": weights["jepa"], **cw, **{n: weights[n] for n in INPUTS}}}


def compute_pnls(ctx: Ctx, strats, preds: dict, params: dict, at: pd.DatetimeIndex, split: str,
                 cost: float, har_fit: pd.DatetimeIndex) -> dict:
    return {s: strategy_pnls(ctx, s, preds, params, at, split, cost, har_fit) for s in strats}


def persist_artifacts(out: Path, stage: str, preds: dict, sps: dict) -> None:
    """Write <stage>_{preds,weights,pnl}.parquet. Call BEFORE computing gates.

    preds: per-seed and seed-averaged predictions (vrp_* for S1, p_* for S2/S3) plus the
    pca/lastday control predictions. weights/pnl: every strategy and comparator.
    """
    out = Path(out)
    cols = {}
    for key, d in preds.items():
        for name, v in d.items():
            if name == "jepa_seeds":
                for k, sr in enumerate(v):
                    cols[f"{key}_jepa_seed_idx{k}"] = sr
            else:
                cols[f"{key}_{name}"] = v
    pd.DataFrame(cols).to_parquet(out / f"{stage}_preds.parquet")
    w, p = {}, {}
    for s, sp in sps.items():
        for n, v in sp["weights"].items():
            w[f"{s}__{n}"] = v
        p[f"{s}__strategy"] = sp["pnl"]
        for n, v in sp["comparators"].items():
            p[f"{s}__{n}"] = v
    pd.DataFrame(w).to_parquet(out / f"{stage}_weights.parquet")
    pd.DataFrame(p).to_parquet(out / f"{stage}_pnl.parquet")


def g1_for(ctx: Ctx, strat: str, preds: dict, at: pd.DatetimeIndex) -> dict:
    """G1 for S1 (squared error on VRP) or S2 (Brier on crash label)."""
    if strat == "S1":
        y, key, loss = ctx.vrp.loc[at], "vrp", sq_error
    else:
        y = ctx.crash_y.loc[at]
        key, loss = "p", brier
    return g1(loss(preds[key]["jepa"], y), [loss(s, y) for s in preds[key]["jepa_seeds"]],
              {n: loss(preds[key][n], y) for n in INPUTS}, min_seeds=ctx.min_seeds)


def gates(ctx: Ctx, strats, preds: dict, params: dict, at: pd.DatetimeIndex, split: str,
          windows: list, cost: float, har_fit: pd.DatetimeIndex, with_g1: bool = True,
          sps: dict = None) -> dict:
    """Per-strategy G1/G2/G3. S3 shares S2's G1 (computed on demand, not reported for S2)."""
    out, g1_cache = {}, {}
    for s in strats:
        sp = sps[s] if sps else strategy_pnls(ctx, s, preds, params, at, split, cost, har_fit)
        r = {"g2": g2(sp["pnl"], sp["comparators"], sp["dd_bench"], sp["strategy_only"]),
             "g3": g3(sp["pnl"], sp["dd_bench"], windows)}
        if "hedge_h" in sp:
            r["g2"]["comparators"]["always_hedged"]["h_notional"] = sp["hedge_h"]
        if with_g1:
            src = "S2" if s == "S3" else s
            if src not in g1_cache:
                g1_cache[src] = g1_for(ctx, src, preds, at)
            r["g1"] = ({"shared_with": "S2", "pass": g1_cache["S2"]["pass"]} if s == "S3"
                       else g1_cache[s])
            r["eligible_for_test"] = bool(r["g1"]["pass"] and r["g2"]["jepa_pass"]
                                          and r["g3"]["pass"])
        out[s] = r
    return out


def require_clean_tree() -> None:
    """Non-smoke val/test must run from committed (tracked-clean) code."""
    if git_dirty():
        raise RuntimeError("working tree has tracked changes; commit before a real val/test run")


def require_no_prior_val(out: Path) -> None:
    if (Path(out) / "val_gate.json").exists():
        raise RuntimeError(f"{Path(out) / 'val_gate.json'} already exists; val is a one-shot "
                           "look and is never re-run")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_inputs(keys, data_dir: Path, out: Path) -> dict:
    """sha256 per key: 'raw/<file>' under data_dir, anything else under out; MISSING if absent."""
    res = {}
    for k in keys:
        p = Path(data_dir) / k[4:] if k.startswith("raw/") else Path(out) / k
        res[k] = sha256_file(p) if p.exists() else "MISSING"
    return res


def input_keys(seeds, smoke: bool) -> list:
    """Files pinned at val time: raw Cboe CSVs + SPX.parquet (real runs) and every encoder."""
    raw = [] if smoke else [f"raw/{s}.csv" for s in CBOE_SYMS] + ["raw/SPX.parquet"]
    return raw + [f"encoder_seed{k}.pt" for k in seeds]


def verify_inputs(recorded: dict, data_dir: Path, out: Path) -> None:
    """Refuse (RuntimeError) if any input differs from the val-time hash, or is missing."""
    if not recorded:
        raise RuntimeError("val_gate.json meta has no input_sha256; cannot pin test inputs")
    now = hash_inputs(recorded, data_dir, out)
    bad = sorted(k for k in recorded if now[k] != recorded[k])
    if bad:
        raise RuntimeError(f"input files changed since the val run (or missing): {bad}")


def truncate_raw(raw: dict, data_end) -> dict:
    """Cut every raw series at the recorded data_end so later data never enters the test."""
    return {k: v.loc[:pd.Timestamp(data_end)] for k, v in raw.items()}


def prelock_checks(ctx: Ctx, data_end, windows: list) -> None:
    """Validity-only checks (no test-data-dependent decisions); raise RuntimeError."""
    if ctx.dates[-1] != pd.Timestamp(data_end):
        raise RuntimeError(f"panel ends {ctx.dates[-1].date()}, val recorded data_end {data_end}")
    if not np.isfinite(ctx.X).all():
        raise RuntimeError("non-finite value in the feature matrix")
    if ctx.spx.loc[:data_end].isna().any() or ctx.vix.loc[ctx.dates[0]:data_end].isna().any():
        raise RuntimeError("NaN in SPX or VIX through data_end")
    if len(ctx.splits["test"]) == 0:
        raise RuntimeError("test split is empty")
    lo, hi = ctx.bounds("test")
    for s, e in windows:
        if not (lo <= pd.Timestamp(s) <= pd.Timestamp(e) <= hi):
            raise RuntimeError(f"stress window {s}:{e} not within test bounds {lo.date()}:{hi.date()}")


def apply_core_intersection(res: dict, strats, core_gate: Path) -> None:
    """Extended may not rescue core: eligible = extended-eligible AND core-eligible."""
    core, note = {}, None
    try:
        core = json.loads(Path(core_gate).read_text())
    except (OSError, ValueError):
        note = f"core val_gate.json missing or unreadable at {core_gate}"
    for s in strats:
        ext = bool(res[s]["eligible_for_test"])
        c = isinstance(core.get(s), dict) and core[s].get("eligible_for_test") is True
        res[s]["eligible_extended_only"] = ext
        res[s]["eligible_for_test"] = bool(ext and c)
        res[s]["eligibility_note"] = note or ("core not eligible" if ext and not c else "ok")


def synthetic_raw(seed: int = 7) -> dict:
    """Network-free raw data with regime-switching vol and stress episodes in every split."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-06-01", "2026-10-30")
    sigma = np.full(len(idx), 0.12)
    episodes = [("2008-09-15", "2009-04-15"), ("2010-05-06", "2010-06-30"),
                ("2011-08-01", "2011-10-15"), ("2013-06-01", "2013-06-30"),
                ("2015-08-17", "2015-10-01"), ("2016-01-04", "2016-03-01"),
                ("2018-02-01", "2018-03-15"), ("2018-10-08", "2018-12-31"),
                ("2020-02-20", "2020-05-15"), ("2022-01-03", "2022-10-14"),
                ("2025-04-01", "2025-05-01")]
    for a, b in episodes:
        sigma[(idx >= a) & (idx <= b)] = 0.45
    sigma *= np.exp(rng.normal(0, 0.15, len(idx)))
    r = sigma / np.sqrt(252) * rng.standard_normal(len(idx)) + 0.0003
    spx = pd.Series(1000 * np.exp(np.cumsum(r)), index=idx, name="spx")
    rv = np.sqrt((pd.Series(r, idx) ** 2 * 252).ewm(span=10).mean())
    vix = (100 * 1.1 * rv + rng.normal(0, 1.0, len(idx))).clip(lower=9.0)
    vix = pd.Series(vix.to_numpy(), idx)
    return {
        "spx": spx,
        "vix": vix,
        "vix6m": vix.ewm(span=60).mean() * 0.7 + 0.3 * 20 + 1.5,
        "vvix": 90 + 0.5 * (vix - 15) + pd.Series(rng.normal(0, 5, len(idx)), idx),
        "skew": 130 + pd.Series(rng.normal(0, 5, len(idx)), idx),
        "vix3m": vix.ewm(span=30).mean() * 0.85 + 3,
        "vix9d": vix.ewm(span=3).mean(),
    }
