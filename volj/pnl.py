"""Laddered short-variance P&L (mark-to-market) and performance metrics.

Timing: weight w_t at signal date t opens notional w_t/h at close t+1 (entry e).
Realized variance accrues over returns e+1..e+h; the position closes at e+h.
"""

import numpy as np
import pandas as pd

from volj.config import H


def short_var_ladder_pnl(weights: pd.Series, spx: pd.Series, vix_pts: pd.Series,
                         cost_volpt: float, h: int = H) -> pd.Series:
    """Daily short-variance P&L on the SPX index, 0.0 on days with no live position.

    Long-var mark on day s (n = s - e): (n/h)(RV2_acc - K2) + ((h-n)/h)(VIX_s^2 - K2),
    K2 = ((VIX_e - cost)/100)^2, RV2_acc = 252/n * sum r^2(e+1..s). Short P&L is
    -notional * change in mark (mark is 0 before entry). Positions whose life runs
    past the data end are marked through the last available day.
    """
    idx = spx.index
    n_days = len(idx)
    w = weights.reindex(idx).fillna(0.0).to_numpy()
    r2 = (np.log(spx).diff() ** 2).to_numpy()
    cum_r2 = np.cumsum(np.nan_to_num(r2))  # NaNs are checked per position below
    vix = vix_pts.reindex(idx).to_numpy()
    vix_var = (vix / 100.0) ** 2

    pnl = np.zeros(n_days)
    for t in range(n_days - 1):
        if w[t] == 0.0:
            continue
        e = t + 1
        last = min(e + h, n_days - 1)
        n = np.arange(0, last - e + 1)
        s = e + n
        for name, bad in (("VIX", np.isnan(vix[e:last + 1])),
                          ("SPX return", np.isnan(r2[e:last + 1]))):
            if bad.any():
                raise ValueError(f"missing {name} on {idx[e + int(np.argmax(bad))].date()}")
        k2 = ((vix[e] - cost_volpt) / 100.0) ** 2
        acc = np.where(n > 0, 252.0 / np.maximum(n, 1) * (cum_r2[s] - cum_r2[e]), 0.0)
        mtm = (n / h) * (acc - k2) + ((h - n) / h) * (vix_var[s] - k2)
        pnl[s] += -(w[t] / h) * np.diff(mtm, prepend=0.0)
    return pd.Series(pnl, index=idx)


def _check_finite(pnl: pd.Series) -> None:
    if pnl.isna().any():
        raise ValueError("pnl contains NaN")


def long_spx_pnl(spx: pd.Series) -> pd.Series:
    """Simple daily SPX returns; first day is 0.0 (P&L is never NaN)."""
    return spx.pct_change().fillna(0.0)


def sharpe(pnl: pd.Series) -> float:
    """Annualized mean/std * sqrt(252); 0.0 if empty or zero variance; raises on NaN."""
    _check_finite(pnl)
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of pnl.cumsum(), peak starting at 0; positive; raises on NaN."""
    _check_finite(pnl)
    cum = pnl.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum.to_numpy()]))[1:]
    return float((peak - cum.to_numpy()).max())


def window_loss(pnl: pd.Series, start: str, end: str) -> float:
    _check_finite(pnl)
    return float(-pnl.loc[start:end].sum())
