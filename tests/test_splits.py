import pandas as pd
import pytest

from volj.splits import split_dates, test_end as _test_end


@pytest.fixture
def idx():
    return pd.bdate_range("2007-06-01", "2026-11-20")


def test_split_embargo(idx):
    tr = split_dates(idx, "train")
    last_le = idx.searchsorted(pd.Timestamp("2016-12-30"), side="right") - 1
    assert idx.get_loc(tr[-1]) == last_le - 22
    va = split_dates(idx, "val")
    assert va[0] >= pd.Timestamp("2017-01-03")
    assert va[0] == idx[idx >= pd.Timestamp("2017-01-03")][0]


def test_no_label_window_crosses_split_end(idx):
    for name, end in [("train", "2016-12-30"), ("val", "2020-12-31"),
                      ("inner_fit", "2014-12-31"), ("inner_select", "2016-12-30")]:
        d = split_dates(idx, name)
        end_pos = idx.searchsorted(pd.Timestamp(end), side="right") - 1
        assert max(idx.get_loc(t) for t in d) + 22 == end_pos


def test_unlabeled_tail_excluded(idx):
    assert idx.get_loc(_test_end(idx)) == len(idx) - 1 - 22


def test_unknown_split(idx):
    with pytest.raises(KeyError, match="bogus"):
        split_dates(idx, "bogus")
