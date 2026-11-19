"""Build an aligned dataset of TWSE-listed companies ranked by market cap."""
import argparse
import json
from datetime import date, datetime
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "TW50" / "tw50.npz"
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TWSE = "https://openapi.twse.com.tw/v1/"


def fetch_twse(endpoint):
    with urlopen(TWSE + endpoint, timeout=30) as response:
        return pd.DataFrame(json.load(response))


def numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def rank_companies(companies, quotes):
    companies = companies.rename(columns={
        "公司代號": "symbol", "公司簡稱": "name", "產業別": "industry",
        "已發行普通股數或TDR原股發行股數": "shares",
    })
    quotes = quotes.rename(columns={
        "Code": "symbol", "ClosingPrice": "close", "Date": "quote_date",
    })
    quotes["quote_date"] = quotes["quote_date"].astype(str)
    quote_date = quotes["quote_date"].max()
    quotes = quotes.loc[quotes["quote_date"] == quote_date, ["symbol", "close"]]
    ranked = companies[["symbol", "name", "industry", "shares"]].merge(
        quotes, on="symbol", validate="one_to_one"
    )
    ranked["shares"] = numeric(ranked["shares"])
    ranked["close"] = numeric(ranked["close"])
    ranked = ranked.loc[
        ranked["symbol"].str.fullmatch(r"[1-9][0-9]{3}")
        & ranked["shares"].gt(0) & ranked["close"].gt(0)
    ].assign(market_cap_twd=lambda rows: rows["shares"] * rows["close"])
    if len(ranked) < 50:
        raise ValueError(f"TWSE snapshot returned only {len(ranked)} eligible companies.")
    ranked = ranked.sort_values(["market_cap_twd", "symbol"], ascending=[False, True]).head(50).copy()
    ranked.insert(0, "rank", range(1, 51))
    ranked["quote_date"] = quote_date
    return ranked


def align_history(history, tickers):
    if history is None or history.empty or "^TWII" not in history.columns.get_level_values(0):
        raise ValueError("Missing TWII history. Retry the Yahoo Finance download.")
    calendar = history["^TWII"]["Close"].dropna().sort_index().index
    if calendar.empty or calendar.has_duplicates:
        raise ValueError("TWII returned an empty calendar or duplicate trading dates.")
    arrays = []
    for ticker in tickers:
        if ticker not in history.columns.get_level_values(0):
            raise ValueError(f"Missing history for {ticker}. Retry or extend the requested range.")
        frame = history[ticker].reindex(index=calendar, columns=FEATURES)
        values = frame.to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] > 0)
        if not valid.any():
            raise ValueError(f"No valid OHLCV for {ticker}. Retry or extend the requested range.")
        values[~valid] = np.nan
        arrays.append(values)
    return calendar.strftime("%Y-%m-%d").to_numpy(dtype=str), np.stack(arrays)


def main():
    import yfinance as yf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, help="Exclusive (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    today = datetime.now(ZoneInfo("Asia/Taipei")).date()
    end = args.end or today
    if args.start >= end or end > today:
        parser.error("Use start < end <= today (end is exclusive).")

    ranked = rank_companies(
        fetch_twse("opendata/t187ap03_L"), fetch_twse("exchangeReport/STOCK_DAY_ALL")
    )
    tickers = [f"{symbol}.TW" for symbol in ranked["symbol"]]
    history = yf.download(
        tickers + ["^TWII"], start=args.start, end=end, interval="1d",
        group_by="ticker", auto_adjust=True, multi_level_index=True, threads=4,
    )
    dates, ohlcv = align_history(history, tickers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.savez_compressed(
            output, ohlcv=ohlcv, dates=dates, symbols=ranked["symbol"].to_numpy(dtype=str),
            features=np.array(FEATURES), quote_date=str(ranked["quote_date"].iloc[0]),
        )
    ranked.to_csv(args.output.with_suffix(".companies.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved {len(tickers)} stocks × {len(dates)} trading days to {args.output}")


if __name__ == "__main__":
    main()
