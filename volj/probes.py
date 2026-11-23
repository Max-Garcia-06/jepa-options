"""Linear probes (ridge on VRP, logistic on crash label) on frozen embeddings."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from volj.config import LOGIT_CS, MIN_CRASH_POSITIVES, RIDGE_ALPHAS

Kind = Literal["ridge", "logit"]


def fit_probe(kind: Kind, X: pd.DataFrame, y: pd.Series, param: float) -> Pipeline:
    """StandardScaler + Ridge(alpha=param) or LogisticRegression(C=param).

    Rows with NaN y are dropped; NaN in X raises ValueError. Logit raises
    ValueError("too few crash positives") if fewer than MIN_CRASH_POSITIVES ones remain.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y indexes differ")
    keep = y.notna().to_numpy()
    Xk, yk = X.loc[keep], y.loc[keep]
    if Xk.isna().any().any():
        raise ValueError("NaN in probe features")
    if kind == "ridge":
        model = Ridge(alpha=param)
    elif kind == "logit":
        if (yk == 1).sum() < MIN_CRASH_POSITIVES:
            raise ValueError("too few crash positives")
        model = LogisticRegression(C=param, max_iter=1000)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return Pipeline([("scale", StandardScaler()), ("model", model)]).fit(Xk.to_numpy(), yk.to_numpy())


def predict(kind: Kind, est: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Ridge value or P(class 1), indexed like X."""
    if kind == "ridge":
        out = est.predict(X.to_numpy())
    else:
        out = est.predict_proba(X.to_numpy())[:, list(est.classes_).index(1)]
    return pd.Series(out, index=X.index)


def sq_error(pred: pd.Series, y: pd.Series) -> pd.Series:
    return (pred - y) ** 2


def brier(p: pd.Series, y: pd.Series) -> pd.Series:
    return (p - y) ** 2


def select_param(kind: Kind, Xf, yf, Xs, ys) -> float:
    """Grid value minimizing select-set loss (ridge MSE / logit Brier).

    Ties resolve to the first value in grid order (np.argmin).
    """
    grid = RIDGE_ALPHAS if kind == "ridge" else LOGIT_CS
    loss_fn = sq_error if kind == "ridge" else brier
    ks = ys.notna().to_numpy()
    losses = []
    for param in grid:
        est = fit_probe(kind, Xf, yf, param)
        pred = predict(kind, est, Xs.loc[ks])
        losses.append(float(loss_fn(pred, ys.loc[ks]).mean()))
    return float(grid[int(np.argmin(losses))])
