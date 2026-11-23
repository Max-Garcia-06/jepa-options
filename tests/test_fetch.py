import pandas as pd
import pytest
import requests

from volj.data.fetch import fetch_cboe, parse_cboe_csv

OHLC = (
    "DATE,OPEN,HIGH,LOW,CLOSE\n"
    "01/02/1990,17.24,17.24,17.24,17.24\n"
    "01/03/1990,18.19,18.19,18.19,18.19\n"
)


def test_parse_ohlc_layout():
    s = parse_cboe_csv(OHLC, "VIX")
    assert s.tolist() == [17.24, 18.19]
    assert list(s.index) == [pd.Timestamp("1990-01-02"), pd.Timestamp("1990-01-03")]
    assert s.name == "vix"
    assert s.index.name == "date"


def test_parse_single_column_layout():
    s = parse_cboe_csv("DATE,VVIX\n03/06/2006,71.73\n", "VVIX")
    assert s.tolist() == [71.73]
    assert s.name == "vvix"


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_html_response_raises(tmp_path, monkeypatch):
    html = "<head><title>307 Temporary Redirect</title></head>"
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(html))
    with pytest.raises(ValueError, match="VIX"):
        fetch_cboe("VIX", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_hit_skips_network(tmp_path, monkeypatch):
    (tmp_path / "VIX.csv").write_text(OHLC)

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(requests, "get", boom)
    s = fetch_cboe("VIX", tmp_path)
    assert s.tolist() == [17.24, 18.19]


def test_duplicate_dates_raise():
    text = "DATE,VIX\n01/02/1990,1.0\n01/02/1990,2.0\n"
    with pytest.raises(ValueError, match="VIX"):
        parse_cboe_csv(text, "VIX")


def test_unsorted_dates_sorted():
    s = parse_cboe_csv("DATE,VIX\n01/03/1990,2.0\n01/02/1990,1.0\n", "VIX")
    assert s.tolist() == [1.0, 2.0]
    assert s.index.is_monotonic_increasing


def _fake_download(monkeypatch):
    import yfinance

    idx = pd.to_datetime(["2026-11-20", "2026-11-23"])
    cols = pd.MultiIndex.from_product([["Close"], ["^GSPC"]])
    df = pd.DataFrame([[5000.0], [5010.0]], index=idx, columns=cols)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: df)


def test_spx_partial_bar_dropped_before_cutoff(tmp_path, monkeypatch):
    from datetime import datetime

    from volj.data.fetch import NY, fetch_spx

    _fake_download(monkeypatch)
    s = fetch_spx(tmp_path, now=datetime(2026, 11, 23, 12, 0, tzinfo=NY))
    assert s.index[-1] == pd.Timestamp("2026-11-20")
    assert s.name == "spx"
    assert pd.read_parquet(tmp_path / "SPX.parquet")["spx"].index[-1] == pd.Timestamp("2026-11-20")


def test_spx_final_bar_kept_after_cutoff(tmp_path, monkeypatch):
    from datetime import datetime

    from volj.data.fetch import NY, fetch_spx

    _fake_download(monkeypatch)
    s = fetch_spx(tmp_path, now=datetime(2026, 11, 23, 16, 30, tzinfo=NY))
    assert s.index[-1] == pd.Timestamp("2026-11-23")
    assert s.tolist() == [5000.0, 5010.0]


def test_spx_cache_hit_filters_partial_bar(tmp_path):
    from datetime import datetime

    from volj.data.fetch import NY, fetch_spx

    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-20", "2026-11-23"]), name="date")
    pd.Series([5000.0, 5010.0], index=idx, name="spx").to_frame().to_parquet(
        tmp_path / "SPX.parquet"
    )
    s = fetch_spx(tmp_path, now=datetime(2026, 11, 23, 10, 0, tzinfo=NY))
    assert s.index[-1] == pd.Timestamp("2026-11-20")
    s = fetch_spx(tmp_path, now=datetime(2026, 11, 23, 17, 0))  # naive = ET
    assert s.index[-1] == pd.Timestamp("2026-11-23")


def _patch_download(monkeypatch, dates, values):
    import yfinance

    cols = pd.MultiIndex.from_product([["Close"], ["^GSPC"]])
    df = pd.DataFrame([[v] for v in values], index=pd.to_datetime(dates), columns=cols)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: df)


def test_spx_duplicate_dates_raise(tmp_path, monkeypatch):
    from volj.data.fetch import fetch_spx

    _patch_download(monkeypatch, ["2026-11-18", "2026-11-18"], [1.0, 2.0])
    with pytest.raises(ValueError, match="spx"):
        fetch_spx(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_spx_unsorted_sorted(tmp_path, monkeypatch):
    from volj.data.fetch import fetch_spx

    _patch_download(monkeypatch, ["2026-11-19", "2026-11-18"], [2.0, 1.0])
    s = fetch_spx(tmp_path)
    assert s.tolist() == [1.0, 2.0]
    assert s.index.is_monotonic_increasing
