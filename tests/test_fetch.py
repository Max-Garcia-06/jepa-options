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
