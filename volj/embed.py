"""Context-encoder embeddings. Uses only data at or before each date."""

import numpy as np
import pandas as pd
import torch

from volj.config import L
from volj.model import ContextEncoder


def window_positions(dates: pd.DatetimeIndex, at: pd.DatetimeIndex, L: int = L) -> np.ndarray:
    """Integer positions of `at` in `dates`; raises if absent or lacking L days of context."""
    pos = dates.get_indexer(at)
    if (pos < 0).any():
        raise ValueError(f"dates not in panel: {list(at[pos < 0][:3])}")
    if (pos < L - 1).any():
        raise ValueError(f"fewer than {L} days of context for: {list(at[pos < L - 1][:3])}")
    return pos


def embed(
    encoder: ContextEncoder,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    at: pd.DatetimeIndex,
    L: int = L,
    batch: int = 512,
) -> pd.DataFrame:
    """Embed the window X[t-L+1..t] for each t in `at`."""
    pos = window_positions(dates, at, L)
    out = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(pos), batch):
            w = np.stack([X[t - L + 1 : t + 1] for t in pos[i : i + batch]])
            out.append(encoder(torch.tensor(w, dtype=torch.float32)).numpy())
    z = np.concatenate(out).astype(np.float64) if out else np.empty((0, 0))
    return pd.DataFrame(z, index=at, columns=[f"z{i}" for i in range(z.shape[1])])
