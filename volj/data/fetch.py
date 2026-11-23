"""Fetch Cboe index histories and SPX closes, with on-disk cache and validation."""

import io
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
CLOSE_CUTOFF = (16, 15)  # ET time after which today's SPX bar is final

CBOE_SYMS = ["VIX", "VIX6M", "VVIX", "SKEW", "VIX3M", "VIX9D"]
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"


def parse_cboe_csv(text: str, sym: str) -> pd.Series:
    """Validate and parse a Cboe history CSV into a close series (raw index points)."""
    text = text.lstrip("\ufeff").lstrip()
    first = text.split("\n", 1)[0].strip()
    if not first.upper().startswith("DATE,"):
        raise ValueError(
            f"Cboe {sym}: response is not a CSV (header {first[:80]!r}); "
            "expected a line starting with 'DATE,'"
        )
    df = pd.read_csv(io.StringIO(text))
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
    if not s.index.is_unique:
        raise ValueError(f"Cboe {sym}: duplicate dates in CSV")
    return s.sort_index()


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


def _drop_partial_bar(s: pd.Series, now: datetime | None) -> pd.Series:
    """Drop rows dated today (ET) unless the ET session close (16:15) has passed."""
    if now is None:
        now = datetime.now(NY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=NY)
    now_et = now.astimezone(NY)
    if (now_et.hour, now_et.minute) > CLOSE_CUTOFF:
        return s
    partial = s.index >= pd.Timestamp(now_et.date())
    if partial.any():
        log.warning(
            "spx: dropping incomplete bar(s) %s",
            [d.date().isoformat() for d in s.index[partial]],
        )
        s = s[~partial]
    return s


def fetch_spx(
    cache_dir: Path, refresh: bool = False, now: datetime | None = None
) -> pd.Series:
    cache_dir = Path(cache_dir)
    path = cache_dir / "SPX.parquet"
    if path.exists() and not refresh:
        return _drop_partial_bar(pd.read_parquet(path)["spx"], now)
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
    if not s.index.is_unique:
        raise ValueError("spx: duplicate dates from yfinance")
    s = _drop_partial_bar(s.sort_index(), now)
    cache_dir.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def load_raw(
    cache_dir: Path, syms: Sequence[str], refresh: bool = False
) -> dict[str, pd.Series]:
    out = {"spx": fetch_spx(cache_dir, refresh)}
    for sym in syms:
        out[sym.lower()] = fetch_cboe(sym, cache_dir, refresh)
    return out
