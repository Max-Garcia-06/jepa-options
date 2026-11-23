"""Controls for the embedding probes (PCA, last-day) and the HAR baseline."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from volj.config import D, H, L
from volj.embed import window_positions
from volj.targets import forward_rv2


def _flat_windows(X: np.ndarray, pos: np.ndarray, L: int) -> np.ndarray:
    return np.stack([X[t - L + 1 : t + 1].ravel() for t in pos])


def pca_control(
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    at: pd.DatetimeIndex,
    fit_on: pd.DatetimeIndex,
    L: int = L,
    d: int = D,
) -> pd.DataFrame:
    """PCA(d) of flattened L-day windows, fit on `fit_on` windows only."""
    pca = PCA(n_components=d, random_state=0)
    pca.fit(_flat_windows(X, window_positions(dates, fit_on, L), L))
    z = pca.transform(_flat_windows(X, window_positions(dates, at, L), L))
    return pd.DataFrame(z, index=at, columns=[f"p{i}" for i in range(d)])


def lastday_control(panel_scaled: pd.DataFrame, at: pd.DatetimeIndex) -> pd.DataFrame:
    """The scaled feature row at t."""
    return panel_scaled.loc[at].copy()


def har_regressors(spx: pd.Series) -> pd.DataFrame:
    """rv2_k = (252/k) * sum of squared log returns over the last k days ending at t."""
    r2 = np.log(spx).diff() ** 2
    return pd.DataFrame(
        {f"rv2_{k}": (252.0 / k) * r2.rolling(k).sum() for k in (1, 5, 22)}
    )


def fit_ols(regs: pd.DataFrame, y: pd.Series, fit_on: pd.DatetimeIndex) -> tuple[float, np.ndarray]:
    """OLS with intercept on `fit_on` dates where regressors and y are finite."""
    df = regs.loc[fit_on].join(y.rename("y").loc[fit_on]).dropna()
    A = np.column_stack([np.ones(len(df)), df[regs.columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    return float(coef[0]), coef[1:]


def har_vrp(
    spx: pd.Series,
    vix_pts: pd.Series,
    fit_on: pd.DatetimeIndex,
    at: pd.DatetimeIndex,
    cost_volpt: float,
) -> pd.Series:
    """K2 at t (VIX at t, net of cost) minus the HAR forecast of forward_rv2, at `at`.

    The signal at t uses only VIX at t (not the t+1 entry strike).

    Contract: `fit_on` must be embargoed. The label at fit date t uses returns up
    to position t+H+1, so the last fit position + H + 1 must be strictly before
    the first out-of-sample `at` position; otherwise ValueError.
    """
    regs = har_regressors(spx)
    pos_at = spx.index.get_indexer(at)
    if (pos_at < 0).any():
        raise ValueError("`at` dates not in spx index")
    if regs.iloc[pos_at].isna().any().any():
        raise ValueError("`at` dates have NaN HAR regressors (insufficient history)")
    oos = at[at > fit_on[-1]]
    if len(oos):
        if spx.index.get_loc(fit_on[-1]) + H + 1 >= spx.index.get_loc(oos[0]):
            raise ValueError("HAR fit labels overlap evaluation dates; pass embargoed fit_on")
    intercept, beta = fit_ols(regs, forward_rv2(spx), fit_on)
    forecast = intercept + regs.loc[at].to_numpy() @ beta
    k2 = ((vix_pts.loc[at] - cost_volpt) / 100.0) ** 2
    return (k2 - pd.Series(forecast, index=at)).rename("har_vrp")
