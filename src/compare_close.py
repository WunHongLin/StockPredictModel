"""Compare observed and forecast next-session adjusted closes for one stock."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from download_tw50 import DEFAULT_DATASET
from predict import StockPredictor


def compare_closes(predictor, symbol, start, end):
    if symbol not in predictor.symbols:
        raise ValueError(f"Stock {symbol} is not in the checkpoint's stock universe.")

    position = predictor.symbols.index(symbol)
    dates = predictor.dates
    first = max(predictor.lookback, int(dates.searchsorted(start)))
    stop = int(dates.searchsorted(end, side="right"))
    rows = []
    for target in range(first, stop):
        signal = target - 1
        forecasts = predictor.predict(signal).set_index("symbol")
        observed = float(predictor.ohlcv[position, target, 3])
        predicted = (
            float(forecasts.at[symbol, "predicted_adjusted_close"])
            if symbol in forecasts.index else np.nan
        )
        rows.append({
            "symbol": symbol,
            "signal_date": str(dates[signal]),
            "target_date": str(dates[target]),
            "actual_adjusted_close": observed if np.isfinite(observed) and observed > 0 else np.nan,
            "predicted_adjusted_close": predicted,
        })

    if not rows:
        raise ValueError("No target sessions in range with enough prior history.")
    result = pd.DataFrame(rows)
    result["error"] = result["predicted_adjusted_close"] - result["actual_adjusted_close"]
    if not result["error"].notna().any():
        raise ValueError("No dates in range have both an actual close and a forecast.")
    return result


def save_chart(result, output, title):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure

    dates = pd.to_datetime(result["target_date"])
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    marker = "." if len(result) < 45 else None
    ax.plot(dates, result["actual_adjusted_close"], label="Actual close",
            color="#146b70", linewidth=2, marker=marker)
    ax.plot(dates, result["predicted_adjusted_close"], label="Predicted close",
            color="#bc4e21", linewidth=1.6, linestyle="--", marker=marker)
    ax.set(title=title, xlabel="Target trading date", ylabel="Adjusted closing price (TWD)")
    locator = AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    with output.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--symbol", required=True, help="Stock code in the dataset, e.g. 2330.")
    parser.add_argument("--start", type=date.fromisoformat, help="First target date, inclusive. Defaults to checkpoint test start.")
    parser.add_argument("--end", type=date.fromisoformat, help="Last target date, inclusive. Defaults to the last data date.")
    parser.add_argument("--output", type=Path, default=Path("close_comparison.png"), help="New PNG path; CSV uses the same stem.")
    args = parser.parse_args()

    try:
        predictor = StockPredictor(args.checkpoint, args.dataset)
        start = args.start or predictor.first_test_date
        if start is None:
            raise ValueError("Older checkpoint: set --start to 'test starts' from the training log for test-only comparison.")
        start = np.datetime64(start, "D")
        end = np.datetime64(args.end, "D") if args.end else predictor.dates[-1]
        if start > end:
            raise ValueError("--start must be on or before --end.")
        if args.output.suffix.lower() != ".png":
            raise ValueError("--output must be a .png path.")
        csv_path = args.output.with_suffix(".csv")
        if args.output.exists() or csv_path.exists():
            raise ValueError("Output PNG or CSV already exists. Choose a new filename.")

        result = compare_closes(predictor, args.symbol, start, end)
        errors = result["error"].dropna()
        mae = errors.abs().mean()
        rmse = np.sqrt(errors.pow(2).mean())
        title = f"{args.symbol} | Actual vs predicted close\nMAE {mae:.2f} TWD | RMSE {rmse:.2f} TWD"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_chart(result, args.output, title)
        result.to_csv(csv_path, index=False, mode="x")
        print(f"Target dates: {result['target_date'].iat[0]} to {result['target_date'].iat[-1]}")
        print(f"Paired sessions: {len(errors)}/{len(result)} | MAE: {mae:.4f} TWD | RMSE: {rmse:.4f} TWD")
        if predictor.first_test_date and result["target_date"].iat[0] < predictor.first_test_date:
            print(f"Includes dates before the checkpoint test period ({predictor.first_test_date}).")
        print(f"Saved {args.output} and {csv_path}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
