"""Pre-registered pass/fail gate, test-set lock, report writer."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from volj.config import ALPHA, BLOCK, BOOT_SEED, MIN_SEEDS_POSITIVE, N_BOOT
from volj.pnl import max_drawdown, sharpe, window_loss

REPO_ROOT = Path(__file__).resolve().parent.parent


def block_bootstrap_ci(d, block: int = BLOCK, n_boot: int = N_BOOT,
                       alpha: float = ALPHA, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Circular block bootstrap of mean(d); two-sided (1-alpha) percentile CI."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or np.isnan(d).any():
        raise ValueError("block_bootstrap_ci needs a non-empty array without NaN")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _aligned_diff(a: pd.Series, b: pd.Series) -> np.ndarray:
    """a - b on the intersection of indexes."""
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        raise ValueError("loss series have no common dates")
    d = a.loc[idx] - b.loc[idx]
    if d.isna().any():
        raise ValueError("NaN in aligned losses")
    return d.to_numpy(dtype=float)


def g1(loss_jepa_avg: pd.Series, loss_jepa_seeds: list[pd.Series],
       loss_controls: dict[str, pd.Series], min_seeds: int = MIN_SEEDS_POSITIVE) -> dict:
    per_control = {}
    ok = True
    for name, lc in loss_controls.items():
        d = _aligned_diff(lc, loss_jepa_avg)
        lo, hi = block_bootstrap_ci(d)
        seeds_pos = int(sum(_aligned_diff(lc, ls).mean() > 0 for ls in loss_jepa_seeds))
        per_control[name] = {"mean_diff": float(d.mean()), "ci": (lo, hi),
                             "seeds_positive": seeds_pos}
        ok = ok and lo > 0 and seeds_pos >= min_seeds
    return {"pass": bool(ok), "per_control": per_control}


def g2(pnl: pd.Series, comparators: dict[str, pd.Series], dd_bench: pd.Series,
       strategy_only: set[str]) -> dict:
    unknown = set(strategy_only) - set(comparators)
    if unknown:
        raise ValueError(f"strategy_only names not in comparators: {sorted(unknown)}")
    sh = sharpe(pnl)
    dd = max_drawdown(pnl)
    dd_ok = dd <= max_drawdown(dd_bench)
    comp = {}
    for name, c in comparators.items():
        csh = sharpe(c)
        comp[name] = {"sharpe": csh, "beats": bool(sh > csh)}
    strat = dd_ok and all(comp[n]["beats"] for n in strategy_only)
    jepa = dd_ok and all(v["beats"] for v in comp.values())
    return {"strategy_pass": bool(strat), "jepa_pass": bool(jepa),
            "sharpe": sh, "max_dd": dd, "comparators": comp}


def g3(pnl: pd.Series, bench: pd.Series, windows: list[tuple[str, str]]) -> dict:
    per = {}
    for s, e in windows:
        pl, bl = window_loss(pnl, s, e), window_loss(bench, s, e)
        per[f"{s}:{e}"] = {"loss": pl, "bench_loss": bl, "pass": bool(pl <= bl)}
    return {"pass": all(v["pass"] for v in per.values()), "windows": per}


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def acquire_test_lock(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    lock = out_dir / "test_look.lock"
    if lock.exists():
        raise RuntimeError("test set already evaluated")
    gate = out_dir / "val_gate.json"
    eligible = False
    if gate.exists():
        data = json.loads(gate.read_text())
        eligible = any(isinstance(v, dict) and v.get("eligible_for_test") is True
                       for v in data.values())
    if not eligible:
        raise RuntimeError("val gate not passed")
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_rev": _git_rev()}
    with open(lock, "x") as f:
        json.dump(payload, f)


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Series):
        return {str(k): _py(v) for k, v in o.items()}
    return o


def write_report(results: dict, path: Path) -> None:
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("report path must end in .json")
    res = _py(results)
    path.write_text(json.dumps(res, indent=2, default=str))
    lines = ["# Gate report", ""]
    for strat, r in res.items():
        lines += [f"## {strat}", ""]
        if not isinstance(r, dict):
            lines += [f"{r}", ""]
            continue
        lines += ["| key | value |", "|---|---|"]
        for k, v in r.items():
            if isinstance(v, dict):
                v = json.dumps(v, default=str)
            lines.append(f"| {k} | {v} |")
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines))
