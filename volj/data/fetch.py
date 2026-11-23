"""Fetch Cboe index histories and SPX closes, with on-disk cache and validation."""

import io
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import requests

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"


def parse_cboe_csv(text: str, sym: str) -> pd.Series:
    """Validate and parse a Cboe history CSV into a close series (raw index points)."""
    first = text.lstrip("﻿").lstrip().split("\n", 1)[0].strip()
    if not first.upper().startswith("DATE,"):
        raise ValueError(
            f"Cboe {sym}: response is not a CSV (header {first[:80]!r}); "
            "expected a line starting with 'DATE,'"
        )
    df = pd.read_csv(io.StringIO(text.lstrip("﻿").lstrip()))
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[1]
    try:
        idx = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
        vals = pd.to_numeric(df[col])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cboe {sym}: could not parse CSV ({e})") from e
    if df.empty:
        raise ValueError(f"Cboe {sym}: CSV has no rows")
    s = pd.Series(vals.to_numpy(dtype=float), index=pd.DatetimeIndex(idx, name="date"))
    s.name = sym.lower()
    return s


def fetch_cboe(sym: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{sym}.csv"
    if path.exists() and not refresh:
        return parse_cboe_csv(path.read_text(), sym)
    resp = requests.get(CBOE_URL.format(sym=sym), timeout=30, allow_redirects=True)
    resp.raise_for_status()
    series = parse_cboe_csv(resp.text, sym)  # validate before caching
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    return series


def fetch_spx(cache_dir: Path, refresh: bool = False) -> pd.Series:
    cache_dir = Path(cache_dir)
    path = cache_dir / "SPX.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["spx"]
    import yfinance

    df = yfinance.download("^GSPC", start="2005-01-01", auto_adjust=False, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # single-ticker MultiIndex columns
        close = close.iloc[:, 0]
    s = close.dropna().astype(float)
    if s.empty:
        raise ValueError("yfinance returned no ^GSPC data")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).tz_localize(None), name="date")
    s.name = "spx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def load_raw(cache_dir: Path, syms: Sequence[str]) -> dict[str, pd.Series]:
    out = {"spx": fetch_spx(cache_dir)}
    for sym in syms:
        out[sym.lower()] = fetch_cboe(sym, cache_dir)
    return out
