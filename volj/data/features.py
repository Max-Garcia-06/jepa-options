"""Feature panel on SPX dates, train-only scaler, and look-ahead guard."""

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from volj.config import FEATURE_SETS, MAX_FFILL

log = logging.getLogger(__name__)

_RV_WINDOWS = (5, 10, 21)


def build_panel(raw: dict[str, pd.Series], feature_set: str) -> pd.DataFrame:
    spx = raw["spx"]
    idx = spx.index
    ret = np.log(spx).diff()
    f = {"ret": ret}
    for n in _RV_WINDOWS:
        f[f"rv{n}"] = np.sqrt(252 / n * (ret**2).rolling(n).sum())

    def lvl(k):
        return raw[k].reindex(idx).ffill(limit=MAX_FFILL)

    vix = lvl("vix") / 100
    f["vix"] = vix
    f["vix6m"] = lvl("vix6m") / 100
    f["vvix"] = lvl("vvix") / 100
    f["skew"] = lvl("skew") / 100
    f["vix_vix6m"] = f["vix"] / f["vix6m"]
    f["vix_rv21"] = vix - f["rv21"]
    if "vix9d" in FEATURE_SETS[feature_set]:
        f["vix9d"] = lvl("vix9d") / 100
        f["vix3m"] = lvl("vix3m") / 100
        f["vix9d_vix"] = f["vix9d"] / vix
        f["vix3m_vix"] = f["vix3m"] / vix

    panel = pd.DataFrame(f, index=idx)[FEATURE_SETS[feature_set]]
    clean = panel.dropna()
    log.info("build_panel(%s): dropped %d of %d rows with NaN",
             feature_set, len(panel) - len(clean), len(panel))
    return clean


def fit_scaler(panel: pd.DataFrame, fit_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    fit = panel.loc[fit_dates]
    return fit.mean(), fit.std()


def apply_scaler(panel: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (panel - mean) / std


def assert_no_lookahead(
    builder: Callable[[dict], pd.DataFrame],
    raw: dict,
    check_dates: Sequence[pd.Timestamp],
) -> None:
    full = builder(raw)
    for d in check_dates:
        trunc = builder({k: s.loc[:d] for k, s in raw.items()})
        if d not in full.index or d not in trunc.index:
            raise AssertionError(f"{d}: row missing in {'full' if d not in full.index else 'truncated'} build")
        for col in full.columns:
            if col not in trunc.columns or not np.isclose(
                full.at[d, col], trunc.at[d, col], rtol=0, atol=1e-12, equal_nan=True
            ):
                raise AssertionError(f"look-ahead in column '{col}' at {d}")
