"""VRP and crash targets, indexed by signal date t (NaN where unavailable).

Timing: signal at close t -> position opened at close t+1 -> P&L over the
21 daily log returns t+2..t+22.
"""

import numpy as np
import pandas as pd

from volj.config import CRASH_Q, H


def forward_rv2(spx: pd.Series, h: int = H) -> pd.Series:
    """252/h * sum(r[t+2..t+h+1]**2), r = daily log returns."""
    r2 = np.log(spx).diff() ** 2
    return (252.0 / h) * r2.rolling(h).sum().shift(-(h + 1))


def entry_k2(vix_pts: pd.Series, cost_volpt: float) -> pd.Series:
    """Variance strike paid at entry (close t+1), net of cost, in decimal var."""
    return ((vix_pts.shift(-1) - cost_volpt) / 100.0) ** 2


def vrp_target(spx: pd.Series, vix_pts: pd.Series, cost_volpt: float) -> pd.Series:
    """Short-variance payoff per unit: entry_k2 - forward_rv2."""
    return entry_k2(vix_pts, cost_volpt) - forward_rv2(spx)


def crash_threshold(vrp: pd.Series, fit_dates) -> float:
    return float(vrp.loc[fit_dates].quantile(CRASH_Q))


def crash_label(vrp: pd.Series, threshold: float) -> pd.Series:
    return (vrp <= threshold).astype(float).where(vrp.notna())
