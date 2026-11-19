"""Backtest StockMixer signals from the prior close, trading next open to close."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from download_tw50 import DEFAULT_DATASET
from predict import StockPredictor


def backtest(predictor, start, end, first_unseen_date, top_k, buy_cost, sell_cost):
    dates = predictor.dates
    first = int(dates.searchsorted(start))
    stop = int(dates.searchsorted(end, side="right"))
    if first >= stop:
        raise ValueError("No trading sessions in the requested date range.")
    if first < predictor.lookback:
        raise ValueError("Not enough trading history before the first backtest session.")

    known_cutoff = predictor.first_test_date
    if known_cutoff is None and first_unseen_date is None:
        raise ValueError(
            "This checkpoint predates split metadata. Pass --first-unseen-date "
            "using 'test starts' from the original training log."
        )
    cutoff = np.datetime64(known_cutoff) if known_cutoff else first_unseen_date
    if first_unseen_date is not None:
        cutoff = max(cutoff, first_unseen_date)
    if dates[first] < cutoff:
        raise ValueError(f"Backtest must start on or after the first unseen date ({cutoff}).")

    buy_factor = 1.0 + buy_cost
    sell_factor = 1.0 - sell_cost
    symbol_positions = {symbol: index for index, symbol in enumerate(predictor.symbols)}
    equity = 1.0
    results = []
    for day in range(first, stop):
        # Select stocks before looking at any execution-day prices.
        picks = predictor.predict(day - 1)["symbol"].head(top_k).tolist()
        positions = [symbol_positions[symbol] for symbol in picks]
        opens = predictor.ohlcv[positions, day, 0]
        filled = np.isfinite(opens) & (opens > 0)
        closes = predictor.ohlcv[positions, day, 3][filled]
        if not (np.isfinite(closes) & (closes > 0)).all():
            raise ValueError(f"Missing exit price for an opened position on {dates[day]}.")

        # Each slot gets 1/K of equity. Unfilled slots retain cash without replacement.
        gross = closes / opens[filled]
        daily_return = float(np.sum(gross * sell_factor / buy_factor - 1.0) / top_k)
        equity *= 1.0 + daily_return
        results.append({
            "signal_date": str(dates[day - 1]),
            "trade_date": str(dates[day]),
            "selected_symbols": " ".join(picks),
            "filled_symbols": " ".join(np.array(picks, dtype=str)[filled]),
            "positions": int(filled.sum()),
            "net_return": daily_return,
            "equity": equity,
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="First execution date, inclusive.")
    parser.add_argument("--end", type=date.fromisoformat, help="Last execution date, inclusive.")
    parser.add_argument(
        "--first-unseen-date", type=date.fromisoformat,
        help="First post-validation trading date (required for older checkpoints).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--buy-cost-bps", type=float, default=0, help="All-in cost for each buy, in basis points.")
    parser.add_argument("--sell-cost-bps", type=float, default=0, help="All-in cost for each sell, in basis points.")
    parser.add_argument("--output", type=Path, help="Optional path for a new daily results CSV.")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    if not (0 <= args.buy_cost_bps < 10_000 and 0 <= args.sell_cost_bps < 10_000):
        parser.error("Costs must be between 0 and 10,000 basis points, exclusive of 10,000.")

    try:
        predictor = StockPredictor(args.checkpoint, args.dataset)
        if args.top_k > len(predictor.symbols):
            raise ValueError("--top-k exceeds the number of stocks in the model.")
        start = np.datetime64(args.start)
        end = np.datetime64(args.end) if args.end else predictor.dates[-1]
        if start > end:
            raise ValueError("--start must be on or before --end.")
        cutoff = np.datetime64(args.first_unseen_date) if args.first_unseen_date else None
        results = backtest(
            predictor, start, end, cutoff, args.top_k,
            args.buy_cost_bps / 10_000, args.sell_cost_bps / 10_000,
        )
        equity = results["equity"].to_numpy()
        peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        traded = results.loc[results["positions"] > 0, "net_return"]
        if args.output:
            results.to_csv(args.output, index=False, mode="x")
        print(f"Execution dates: {results['trade_date'].iat[0]} to {results['trade_date'].iat[-1]}")
        print(f"Sessions: {len(results)} (with fills: {len(traded)})")
        print(f"Total compounded net return: {equity[-1] - 1:.2%}")
        print(f"Maximum drawdown: {np.max(1 - equity / peak):.2%}")
        if not traded.empty:
            print(f"Profitable sessions with fills: {(traded > 0).mean():.2%}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
