"""Position rules. All weights are indexed by signal date t (no shifting here)."""

import numpy as np
import pandas as pd

from volj.config import S1_CLIP, S3_H_GRID, S3_TAU_GRID
from volj.pnl import long_spx_pnl, sharpe, short_var_ladder_pnl


def s1_weights(pred_vrp: pd.Series, target_std: float) -> pd.Series:
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError(f"target_std must be finite and > 0, got {target_std}")
    return (pred_vrp / target_std).clip(*S1_CLIP)


def s2_weights(p_crash: pd.Series) -> pd.Series:
    return 1.0 - p_crash


def always_short(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index)


def term_structure(vix: pd.Series, vix6m: pd.Series) -> pd.Series:
    """1.0 if vix < vix6m else 0.0 at t (ratio-invariant to point/100 units)."""
    v6 = vix6m.reindex(vix.index)
    if vix.isna().any() or v6.isna().any():
        raise ValueError("NaN in term-structure inputs")
    return (vix < v6).astype(float)


def _hedged_pnl(hedge_w: pd.Series, h_notional: float, spx, vix_pts, cost_volpt) -> pd.Series:
    """Long SPX plus a long-variance hedge (= minus short-variance ladder)."""
    return long_spx_pnl(spx) - h_notional * short_var_ladder_pnl(hedge_w, spx, vix_pts, cost_volpt)


def s3_pnl(p_crash: pd.Series, tau: float, h_notional: float, spx, vix_pts,
           cost_volpt) -> pd.Series:
    if p_crash.isna().any():
        raise ValueError("NaN in p_crash")
    return _hedged_pnl((p_crash > tau).astype(float), h_notional, spx, vix_pts, cost_volpt)


def always_hedged(index: pd.Index, h_notional: float, spx, vix_pts, cost_volpt) -> pd.Series:
    """S3 P&L with tau = -1: hedge on every date in index."""
    return _hedged_pnl(always_short(index), h_notional, spx, vix_pts, cost_volpt)


def select_s3(p_select: pd.Series, spx, vix_pts, cost) -> tuple[float, float]:
    """(tau, h) maximizing Sharpe of the S3 P&L over p_select's date range.

    Ties resolve to first in grid order (tau-major, then h).
    """
    if len(p_select) == 0:
        raise ValueError("p_select is empty")
    lo, hi = p_select.index[0], p_select.index[-1]
    best, best_sh = None, -np.inf
    for tau in S3_TAU_GRID:
        for h in S3_H_GRID:
            sh = sharpe(s3_pnl(p_select, tau, h, spx, vix_pts, cost).loc[lo:hi])
            if sh > best_sh:
                best, best_sh = (float(tau), float(h)), sh
    return best
