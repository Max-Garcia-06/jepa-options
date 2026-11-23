"""Embargoed split dates.

A signal date t has a label window ending at position pos(t)+h+1; t belongs to
a split only if that window ends on or before the split's last date. This is
the embargo: no label window crosses a split boundary.
"""

import pandas as pd

from volj.config import H, INNER, SPLITS


def split_dates(index: pd.DatetimeIndex, name: str, h: int = H) -> pd.DatetimeIndex:
    bounds = {**SPLITS, **INNER}
    if name not in bounds:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(bounds)}")
    start, end = bounds[name]
    first = int(index.searchsorted(pd.Timestamp(start), side="left"))
    if end is None:
        end_pos = len(index) - 1
    else:
        end_pos = int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    last = end_pos - (h + 1)
    return index[first:last + 1] if last >= first else index[:0]


def test_end(index: pd.DatetimeIndex, h: int = H) -> pd.Timestamp:
    return split_dates(index, "test", h)[-1]
