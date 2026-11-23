"""Check the variance-swap P&L proxy against real SPY option prices (Alpaca, data calls only).

Real ladder: each day a short ATM SPY straddle (expiry nearest 30 calendar days) is opened
at the close and delta-hedged daily with SPY shares for H trading days. Network is used only
in `fetch_straddle`, `fetch_spy_close` and the CLI; no orders are ever placed.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from volj.config import ALPACA_CHECK_START, COST_VOLPT, DIV_YIELD, H, RATE

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def _d1d2(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S, K, T, r, q, sigma, kind: Literal["call", "put"]) -> float:
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, q, sigma, kind: Literal["call", "put"]) -> float:
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return math.exp(-q * T) * norm.cdf(d1)
    return -math.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma) -> float:
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def straddle_iv(price, S, K, T, r=RATE, q=DIV_YIELD) -> float:
    """Implied vol of a call+put straddle price (brentq on [0.01, 5]); raises ValueError outside that range."""
    def f(sig):
        return bs_price(S, K, T, r, q, sig, "call") + bs_price(S, K, T, r, q, sig, "put") - price

    lo, hi = f(0.01), f(5.0)
    if lo >= 0:
        raise ValueError(f"straddle price below 1% vol value ({price:.4f} <= {price + lo:.4f}; "
                         f"S={S}, K={K}, T={T:.4f})")
    if hi <= 0:
        raise ValueError(f"straddle price {price:.4f} exceeds the IV=500% value "
                         f"(S={S}, K={K}, T={T:.4f})")
    return float(brentq(f, 0.01, 5.0, xtol=1e-12))


def hedged_short_straddle_pnl(call_close: pd.Series, put_close: pd.Series, spy_close: pd.Series,
                              K: float, expiry: pd.Timestamp, r=RATE, q=DIV_YIELD,
                              hold: int = H) -> pd.Series:
    """Daily $/share P&L of a short straddle opened at the first close of spy_close.

    Each close the straddle's net delta (call + put, from that day's straddle IV) is hedged
    with +(d_call + d_put) SPY shares (short options carry -(d_call + d_put)). P&L on day i
    = -(straddle_i - straddle_{i-1}) + shares_{i-1} * (S_i - S_{i-1}), for up to `hold`
    days, stopping at expiry. A missing option bar on any window day raises ValueError.
    """
    dates = spy_close.index[spy_close.index <= expiry][:hold + 1]
    if len(dates) < 2:
        raise ValueError("window has fewer than 2 SPY closes")
    c, p, s = (x.reindex(dates) for x in (call_close, put_close, spy_close))
    for name, x in (("call", c), ("put", p)):
        if x.isna().any():
            raise ValueError(f"missing {name} bar on {dates[int(np.argmax(x.isna().to_numpy()))].date()}")
    straddle = (c + p).to_numpy()
    S = s.to_numpy()
    pnl = np.zeros(len(dates) - 1)
    shares = 0.0
    for i in range(len(dates)):
        if i > 0:
            pnl[i - 1] = -(straddle[i] - straddle[i - 1]) + shares * (S[i] - S[i - 1])
        if i < len(dates) - 1:
            T = (expiry - dates[i]).days / 365.0
            sig = straddle_iv(straddle[i], S[i], K, T, r, q)
            shares = bs_delta(S[i], K, T, r, q, sig, "call") + bs_delta(S[i], K, T, r, q, sig, "put")
    return pd.Series(pnl, index=dates[1:])


def vega_scale(S, K, T, price, r=RATE, q=DIV_YIELD) -> float:
    """Factor converting straddle $ P&L to variance units: 2*sigma / straddle vega.

    A variance notional N has vega 2*sigma*N, so a straddle with vega V matches N = V/(2 sigma).
    """
    sig = straddle_iv(price, S, K, T, r, q)
    v = 2.0 * bs_vega(S, K, T, r, q, sig)
    return 2.0 * sig / v


def proxy_agreement(real: pd.Series, proxy: pd.Series) -> dict:
    both = pd.concat([real, proxy], axis=1, join="inner").dropna()
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    return {"daily_corr": corr,
            "cum_sign_agree": bool(np.sign(a.sum()) == np.sign(b.sum())),
            "n_days": int(len(a))}


def ols_beta(real: pd.Series, proxy: pd.Series) -> float:
    """Slope of real on proxy (with intercept)."""
    both = pd.concat([real, proxy], axis=1, join="inner").dropna()
    x, y = both.iloc[:, 1].to_numpy(), both.iloc[:, 0].to_numpy()
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def occ_symbol(expiry: pd.Timestamp, kind: Literal["call", "put"], strike: float, root="SPY") -> str:
    return f"{root}{expiry:%y%m%d}{'C' if kind == 'call' else 'P'}{int(round(strike * 1000)):08d}"


def require_weights_unlocked(out_dir: Path) -> None:
    """Model weights over the check window touch the held-out test period: lock required."""
    if not (Path(out_dir) / "test_look.lock").exists():
        raise RuntimeError(f"--weights-file needs {Path(out_dir) / 'test_look.lock'} "
                           "(test-period look not unlocked); use --weights always_short")


def load_keys() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]


def _daily_closes(bars, symbol) -> pd.Series:
    rows = bars.data.get(symbol, [])
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([b.close for b in rows],
                  index=pd.DatetimeIndex([b.timestamp for b in rows]))
    s.index = pd.DatetimeIndex(s.index.tz_convert("America/New_York").tz_localize(None).normalize())
    return s.sort_index()


def fetch_spy_close(start: pd.Timestamp, end: pd.Timestamp, keys) -> pd.Series:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(*keys)
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day, start=start.to_pydatetime(),
        end=end.to_pydatetime(), adjustment="raw"))
    return _daily_closes(bars, "SPY")


def _candidate_expiries(entry: pd.Timestamp, keys, spot: float) -> list[pd.Timestamp]:
    """Listed SPY expiries (active + inactive) in [entry+28d, entry+40d], nearest entry+30d first.

    Falls back to the nearest weekday to the target if the listing returns nothing.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetOptionContractsRequest

    target = entry + pd.Timedelta(days=30)
    client = TradingClient(*keys, paper=True)
    found: set[pd.Timestamp] = set()
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        token = None
        while True:
            resp = client.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=["SPY"], status=status,
                expiration_date_gte=(target - pd.Timedelta(days=2)).date(),
                expiration_date_lte=(target + pd.Timedelta(days=10)).date(),
                strike_price_gte=f"{spot * 0.95:.2f}", strike_price_lte=f"{spot * 1.05:.2f}",
                limit=1000, page_token=token))
            found |= {pd.Timestamp(c.expiration_date) for c in resp.option_contracts}
            token = resp.next_page_token
            if not token:
                break
    if found:
        return sorted(found, key=lambda d: (abs((d - target).days), d))
    t = target
    while t.weekday() >= 5:
        t -= pd.Timedelta(days=1)
    return [t]


def fetch_straddle(entry: pd.Timestamp, keys, spy_close: pd.Series | None = None,
                   hold: int = H) -> tuple[pd.Series, pd.Series, float, pd.Timestamp]:
    """(call_close, put_close, strike, expiry) for the SPY ATM straddle opened at `entry`.

    Expiry: the listed SPY expiry nearest 30 calendar days that has both bars on `entry`
    and at least `hold` trading days to expiry (the nearest one is sometimes a newly listed
    expiry with no history yet). Strike = SPY's
    entry close rounded to $1. Bars are fetched by OCC symbol, which works for expired contracts.
    """
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    entry = pd.Timestamp(entry).normalize()
    if spy_close is None:
        spy_close = fetch_spy_close(entry - pd.Timedelta(days=7), entry + pd.Timedelta(days=1), keys)
    if entry not in spy_close.index:
        raise ValueError(f"no SPY close on {entry.date()}")
    strike = float(round(spy_close[entry]))
    client = OptionHistoricalDataClient(*keys)
    for expiry in _candidate_expiries(entry, keys, strike):
        n_days = int(((spy_close.index > entry) & (spy_close.index <= expiry)).sum())
        if n_days < hold and spy_close.index[-1] >= expiry:
            continue  # expires before the ladder's hold period ends
        syms = {k: occ_symbol(expiry, k, strike) for k in ("call", "put")}
        bars = client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=list(syms.values()), timeframe=TimeFrame.Day,
            start=entry.to_pydatetime(), end=(expiry + pd.Timedelta(days=1)).to_pydatetime()))
        call, put = _daily_closes(bars, syms["call"]), _daily_closes(bars, syms["put"])
        if entry in call.index and entry in put.index:
            return call, put, strike, expiry
    raise ValueError(f"no SPY {strike:.0f} straddle with bars on entry {entry.date()}")


def run_check(weights: pd.Series, spx, vix, spy_close, keys, cost_volpt=COST_VOLPT, h=H):
    """Real laddered P&L (variance units) and proxy P&L restricted to the used entries."""
    from volj.pnl import short_var_ladder_pnl

    idx = spx.index
    pos = {d: i for i, d in enumerate(idx)}
    real = pd.Series(0.0, index=idx)
    used = pd.Series(0.0, index=idx)
    n_skipped = 0
    for t, w in weights[weights != 0].items():
        e = idx[pos[t] + 1]
        try:
            call, put, K, expiry = fetch_straddle(e, keys, spy_close, h)
            pnl = hedged_short_straddle_pnl(call, put, spy_close.loc[e:], K, expiry, hold=h)
            if len(pnl) < h:
                raise ValueError(f"only {len(pnl)} of {h} days before expiry/data end")
            T0 = (expiry - e).days / 365.0
            scale = vega_scale(spy_close[e], K, T0, call[e] + put[e])
        except ValueError as err:
            log.warning("skip entry %s (signal %s): %s", e.date(), t.date(), err)
            n_skipped += 1
            continue
        pnl = pnl.reindex(idx).dropna()
        real = real.add(pnl * scale * (w / h), fill_value=0.0)
        used[t] = w
    proxy = short_var_ladder_pnl(used, spx, vix, cost_volpt, h)
    return real, proxy, int((used != 0).sum()), n_skipped


def result_json(res: dict) -> str:
    """JSON text with NaN/inf metrics as null (never emits NaN)."""
    clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
             for k, v in res.items()}
    return json.dumps(clean, indent=2, allow_nan=False)


def main(argv=None) -> None:
    from volj.config import FEATURE_SETS
    from volj.data.fetch import load_raw
    from volj.splits import test_end

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["always_short"], default="always_short")
    ap.add_argument("--weights-file", type=Path, default=None,
                    help="parquet of a weights Series indexed by date; needs test_look.lock")
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="core")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out_dir = REPO_ROOT / "outputs" / "r01" / a.feature_set
    out = a.out or out_dir / "alpaca_check.json"
    if a.weights_file is not None:
        require_weights_unlocked(out_dir)

    raw = load_raw(REPO_ROOT / "data" / "raw", ["VIX"])
    spx = raw["spx"]
    vix = raw["vix"].reindex(spx.index).ffill(limit=2)
    end = test_end(spx.index)
    start = pd.Timestamp(ALPACA_CHECK_START)
    sig_dates = spx.index[(spx.index >= start) & (spx.index <= end)]
    if a.weights_file is not None:
        w = pd.read_parquet(a.weights_file).iloc[:, 0].reindex(sig_dates).fillna(0.0)
        mode = f"file:{a.weights_file.name}"
    else:
        w = pd.Series(1.0, index=sig_dates)
        mode = "always_short"
    w = w.reindex(spx.index).fillna(0.0)

    keys = load_keys()
    spy = fetch_spy_close(start - pd.Timedelta(days=7), spx.index[-1] + pd.Timedelta(days=2), keys)
    real, proxy, n_used, n_skipped = run_check(w, spx, vix, spy, keys)
    live = (real != 0) | (proxy != 0)
    real, proxy = real[live], proxy[live]
    agree = proxy_agreement(real, proxy)
    res = {"weights_mode": mode, "start": str(start.date()), "end": str(end.date()),
           "n_entries": n_used, "n_skipped": n_skipped, "daily_corr": agree["daily_corr"],
           "cum_sign_agree": agree["cum_sign_agree"], "beta": ols_beta(real, proxy),
           "n_days": agree["n_days"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    text = result_json(res)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
