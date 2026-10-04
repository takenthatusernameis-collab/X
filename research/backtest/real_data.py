"""Real-data loader for the collected OHLCV universe.

Reads per-ticker CSVs from research/data/raw/ (as recorded in
research/data/manifest.json) and returns BarSequences with consecutive
integer trading-day dates (1..n), plus a date-index mapping for reporting.
This preserves the synthetic toolkit's convention that bar dates are
consecutive ints, while keeping the actual calendar dates available.

Research/simulation only. Not live trading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import BarSequence

COLLECTION_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = COLLECTION_ROOT / "raw"
MANIFEST_PATH = COLLECTION_ROOT / "manifest.json"


def _read_ticker_csv(ticker: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str],
]:
    """Load one ticker CSV; returns (open,high,low,close,volume) arrays and date strings."""
    path = RAW_DIR / f"{ticker}_daily.csv"
    dates: List[str] = []
    opens, highs, lows, closes, volumes = [], [], [], [], []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        assert header == ["date", "open", "high", "low", "close", "adjclose", "volume"], header
        for line in fh:
            if not line.strip():
                continue
            p = line.split(",")
            dates.append(p[0])
            opens.append(float(p[1]))
            highs.append(float(p[2]))
            lows.append(float(p[3]))
            closes.append(float(p[4]))
            volumes.append(float(p[6]))
    return (
        np.asarray(opens, dtype=np.float64),
        np.asarray(highs, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        np.asarray(closes, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        dates,
    )


def load_ticker(ticker: str) -> Tuple[BarSequence, List[str]]:
    """Load one ticker's adjusted-close series from the collected universe.

    Returns:
        (bars, dates): bars is a BarSequence with consecutive int dates
            (1..n); dates maps each bar index (1-based) to its
            "YYYY-MM-DD" calendar date string.
    """
    opens, highs, lows, closes, volumes, dates = _read_ticker_csv(ticker)
    n = len(dates)
    assert n > 0, f"no bars loaded for {ticker}"
    int_dates = np.arange(1, n + 1, dtype=np.int64)
    bars = BarSequence(int_dates, opens, highs, lows, closes, volumes)
    return bars, dates


def load_universe(tickers: Optional[List[str]] = None) -> Dict[str, Tuple[BarSequence, List[str]]]:
    """Load the full collected universe (or a subset).

    Args:
        tickers: explicit subset; if None, use the universe in
            research/data/manifest.json.

    Returns:
        dict ticker -> (BarSequence, date-mapping).
    """
    if tickers is None:
        with open(MANIFEST_PATH) as fh:
            manifest = json.load(fh)
        tickers = manifest["universe"]
    return {ticker: load_ticker(ticker) for ticker in tickers}
