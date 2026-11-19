"""Plot adjusted closes for rolling one-day or fixed-origin 20-day forecasts."""
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


def compare_month(predictor, symbol, month_start):
    if predictor.horizon != 20:
        raise ValueError("--month requires a checkpoint trained with --horizon 20.")
    if symbol not in predictor.symbols:
        raise ValueError(f"Stock {symbol} is not in the checkpoint's stock universe.")

    first_target = int(predictor.dates.searchsorted(month_start))
    as_of = first_target - 1
    previous_month = month_start.astype("datetime64[M]") - 1
    if as_of < 0 or predictor.dates[as_of].astype("datetime64[M]") != previous_month:
        raise ValueError("Dataset needs trading history from the month immediately before --month.")
    base_close, price_ratios = predictor.predict_path(as_of)
    position = predictor.symbols.index(symbol)
    forecast = base_close[position] * price_ratios[position]
    if not np.isfinite(forecast).any():
        raise ValueError(f"No valid forecast for {symbol} from {predictor.dates[as_of]}.")

    known_dates = predictor.dates[first_target:first_target + predictor.horizon]
    observed = predictor.ohlcv[position, first_target:first_target + predictor.horizon, 3]
    actual = np.full(predictor.horizon, np.nan)
    actual[:len(observed)] = np.where(np.isfinite(observed) & (observed > 0), observed, np.nan)
    return pd.DataFrame({
        "symbol": symbol,
        "signal_date": str(predictor.dates[as_of]),
        "target_date": [str(day) for day in known_dates] + [""] * (predictor.horizon - len(known_dates)),
        "horizon": range(1, predictor.horizon + 1),
        "actual_adjusted_close": actual,
        "predicted_adjusted_close": forecast,
        "error": forecast - actual,
    })


def save_chart(result, output, title):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure

    monthly = "horizon" in result
    x = result["horizon"] if monthly else pd.to_datetime(result["target_date"])
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    marker = "." if len(result) < 45 else None
    if result["actual_adjusted_close"].notna().any():
        ax.plot(x, result["actual_adjusted_close"], label="Actual close",
                color="#146b70", linewidth=2, marker=marker)
    ax.plot(x, result["predicted_adjusted_close"], label="Predicted close",
            color="#bc4e21", linewidth=1.6, linestyle="--", marker=marker)
    xlabel = "Trading sessions after signal date" if monthly else "Target trading date"
    ax.set(title=title, xlabel=xlabel, ylabel="Adjusted closing price (TWD)")
    if monthly:
        ax.set_xticks([1, 5, 10, 15, 20])
    else:
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
    parser.add_argument("--month", help="Fixed-origin forecast from the start of YYYY-MM, for a 20-day checkpoint.")
    parser.add_argument("--start", type=date.fromisoformat, help="First target date, inclusive. Defaults to checkpoint test start.")
    parser.add_argument("--end", type=date.fromisoformat, help="Last target date, inclusive. Defaults to the last data date.")
    parser.add_argument("--output", type=Path, default=Path("close_comparison.png"), help="New PNG path; CSV uses the same stem.")
    args = parser.parse_args()

    try:
        predictor = StockPredictor(args.checkpoint, args.dataset)
        if args.output.suffix.lower() != ".png":
            raise ValueError("--output must be a .png path.")
        csv_path = args.output.with_suffix(".csv")
        if args.output.exists() or csv_path.exists():
            raise ValueError("Output PNG or CSV already exists. Choose a new filename.")

        if args.month:
            if args.start or args.end:
                raise ValueError("--month cannot be combined with --start or --end.")
            month_start = date.fromisoformat(args.month + "-01")
            if args.month != month_start.strftime("%Y-%m"):
                raise ValueError("--month must use YYYY-MM.")
            result = compare_month(predictor, args.symbol, np.datetime64(month_start))
            signal_date = result["signal_date"].iat[0]
            title = f"{args.symbol} | 20-session forecast from {signal_date}"
            print(f"Month anchor: {args.month}-01 | last input session: {signal_date}")
        else:
            if predictor.horizon != 1:
                raise ValueError("Use --month YYYY-MM for a 20-day checkpoint.")
            start = args.start or predictor.first_test_date
            if start is None:
                raise ValueError("Older checkpoint: set --start to 'test starts' from the training log for test-only comparison.")
            start = np.datetime64(start, "D")
            end = np.datetime64(args.end, "D") if args.end else predictor.dates[-1]
            if start > end:
                raise ValueError("--start must be on or before --end.")
            result = compare_closes(predictor, args.symbol, start, end)
            title = f"{args.symbol} | Actual vs predicted close"

        errors = result["error"].dropna()
        if not errors.empty:
            mae = errors.abs().mean()
            rmse = np.sqrt(errors.pow(2).mean())
            title += f"\nMAE {mae:.2f} TWD | RMSE {rmse:.2f} TWD"
            print(f"Paired sessions: {len(errors)}/{len(result)} | MAE: {mae:.4f} TWD | RMSE: {rmse:.4f} TWD")
        known_dates = result.loc[result["target_date"].ne(""), "target_date"]
        if not known_dates.empty:
            print(f"Known target dates: {known_dates.iat[0]} to {known_dates.iat[-1]}")
            if predictor.first_test_date and known_dates.iat[0] < predictor.first_test_date:
                print(f"Includes dates before the checkpoint test period ({predictor.first_test_date}).")
        if args.month and len(known_dates) < predictor.horizon:
            print("Future dates absent from the dataset are shown by session number only.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_chart(result, args.output, title)
        result.to_csv(csv_path, index=False, mode="x")
        print(f"Saved {args.output} and {csv_path}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
