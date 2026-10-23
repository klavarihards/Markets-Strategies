"""Download research data from Yahoo Finance and FRED, then freeze it.

Run this in Colab (the development sandbox has no internet access to these
sites). It is the only code in the package that touches the network; every
other module reads the frozen snapshot it produces.

What is downloaded:

- Adjusted daily closes of the research proxies (SPY, EWT, ...). "Adjusted"
  means dividends are added back as if reinvested, so returns computed from
  these prices are total returns, not just price changes.
- Taiwan-listed implementation funds (e.g. 0050.TW), kept for later checks
  of how closely they track the proxies.
- From FRED: the 3-month T-bill rate (cash) and the USD/TWD exchange rate.
"""

from __future__ import annotations

import io
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .snapshot import write_snapshot

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download_prices(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, from Yahoo Finance."""
    import yfinance as yf  # imported here so the rest of the package works without it

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close[[t for t in tickers if t in close.columns]]
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise RuntimeError(f"Yahoo returned no data for: {missing}")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close.sort_index().astype(float)


def download_fred(series: dict[str, str], start: str, end: str | None = None) -> pd.DataFrame:
    """FRED series as columns named by their role (e.g. ``cash_rate``).

    FRED marks missing days (holidays) with ".", which become NaN here; the
    loader forward-fills them onto the trading calendar.
    """
    cols = {}
    for role, series_id in series.items():
        with urllib.request.urlopen(FRED_CSV.format(series=series_id), timeout=60) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        cols[role] = s
    out = pd.DataFrame(cols).loc[start:end]
    out.index.name = "date"
    return out.sort_index()


def build_snapshot(cfg: dict[str, Any], out_dir: str | Path | None = None,
                   end: str | None = None) -> Path:
    """Download everything the config asks for and write a dated snapshot.

    The snapshot goes to ``data/snapshots/<download date>/``. Point
    ``data.snapshot_dir`` in the config at it (or copy it to ``latest``) and
    never edit it afterwards: a new download means a new snapshot folder.
    """
    dcfg = cfg["data"]
    today = date.today().isoformat()
    # Yahoo's `end` is exclusive: ending on today keeps only completed trading days.
    # Without it, a download during market hours would freeze a half-finished day.
    end = end or dcfg.get("end") or today
    out_dir = Path(out_dir) if out_dir else Path("data/snapshots") / today
    prices = download_prices(dcfg["tickers"], dcfg["start"], end)
    taiwan = download_prices(dcfg["taiwan_listed"], dcfg["start"], end)
    fred = download_fred(dcfg["fred"], dcfg["start"], end)
    meta = {
        "source": "yfinance (auto_adjust=True) and FRED CSV",
        "downloaded_on": today,
        "requested_start": dcfg["start"],
        "requested_end": end,
        "tickers": dcfg["tickers"],
        "taiwan_listed": dcfg["taiwan_listed"],
        "fred_series": dcfg["fred"],
    }
    return write_snapshot({"prices": prices, "taiwan_listed": taiwan, "fred": fred}, out_dir, meta)
