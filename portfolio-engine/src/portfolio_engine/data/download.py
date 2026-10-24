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

import numpy as np
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


def build_macro_snapshot(cfg: dict[str, Any], out_dir: str | Path | None = None,
                         end: str | None = None) -> Path:
    """Download the macro series used by the regime models (e.g. credit spreads) into their own snapshot.

    Kept separate from the market-data snapshot, so adding a macro series
    never changes the prices every earlier result was built on.
    """
    mcfg = cfg["macro"]
    today = date.today().isoformat()
    end = end or today
    out_dir = Path(out_dir) if out_dir else Path("data/macro") / today
    fred = download_fred(mcfg["fred"], mcfg.get("start", cfg["data"]["start"]), end)
    meta = {"source": "FRED CSV", "downloaded_on": today, "fred_series": mcfg["fred"], "requested_end": end}
    return write_snapshot({"macro": fred}, out_dir, meta)


CBC_API = "https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName={code}"
# The CBC site rejects requests without a browser-like user agent.
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def download_cbc(code: str) -> pd.DataFrame:
    """One table from the Central Bank of the Republic of China (Taiwan) statistical database API.

    Monthly tables come back with periods like ``2026M08``; they are indexed by
    the last day of that month. Missing values ("-") become NaN.
    """
    import json
    import ssl
    import time

    # Certificates are still fully verified (chain and host name). Only Python 3.13's extra
    # RFC 5280 strictness flag is relaxed: the CBC certificate lacks a Subject Key Identifier,
    # which that flag rejects, while browsers, curl and Python 3.12 (Colab) accept it.
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
    req = urllib.request.Request(CBC_API.format(code=code), headers={"User-Agent": _UA})
    for attempt in range(5):                     # the CBC server drops connections now and then
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                payload = json.loads(resp.read())
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    cols = [c["data"] for c in payload["data"]["structure"]["Table1"]]
    rows = payload["data"]["dataSets"]
    df = pd.DataFrame([r[1:1 + len(cols)] for r in rows], index=[r[0] for r in rows], columns=cols)
    df = df.replace("-", np.nan).astype(float)
    idx = pd.Index(df.index)
    if idx.str.contains("M").all():
        df.index = pd.PeriodIndex(idx.str.replace("M", "-"), freq="M").to_timestamp(how="end").normalize()
    else:
        df.index = pd.to_datetime(idx, format="%Y%m%d")
    df.index.name = "date"
    return df


def build_taiwan_snapshot(cfg: dict[str, Any], out_dir: str | Path | None = None,
                          end: str | None = None) -> Path:
    """Taiwan-listed fund prices (Yahoo) and Taiwan interest rates (CBC), frozen in their own snapshot.

    Kept separate from the research snapshot so adding Taiwan data never
    changes the prices earlier results were built on.
    """
    tcfg = cfg["taiwan"]
    today = date.today().isoformat()
    end = end or today
    out_dir = Path(out_dir) if out_dir else Path("data/taiwan") / today
    funds = download_prices(list(tcfg["funds"].values()), tcfg.get("start", "2014-01-01"), end)
    frames = {"funds": funds}
    for name, code in tcfg["cbc_tables"].items():
        frames[name] = download_cbc(code)
    meta = {"source": "yfinance (auto_adjust=True) and CBC Statistical Database API", "downloaded_on": today,
            "funds": tcfg["funds"], "cbc_tables": tcfg["cbc_tables"], "requested_end": end}
    return write_snapshot(frames, out_dir, meta)
