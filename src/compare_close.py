"""Plot adjusted closes for rolling one-day or fixed-origin multi-session forecasts."""
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


def month_signal_index(predictor, month_start):
    first_target = int(predictor.dates.searchsorted(month_start))
    as_of = first_target - 1
    previous_month = month_start.astype("datetime64[M]") - 1
    if as_of < 0 or predictor.dates[as_of].astype("datetime64[M]") != previous_month:
        raise ValueError("Dataset needs trading history from the month immediately before --month.")
    return as_of


def compare_month(predictor, symbol, month_start):
    if symbol not in predictor.symbols:
        raise ValueError(f"Stock {symbol} is not in the checkpoint's stock universe.")
    as_of = month_signal_index(predictor, month_start)
    first_target = as_of + 1
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


def check_independent_models(predictors, symbol):
    if any(p.horizon != 1 for p in predictors):
        raise ValueError("Independent comparisons require one-output checkpoints trained with --horizon 1.")
    if len({p.target_offset for p in predictors}) != len(predictors):
        raise ValueError("Independent comparisons require one checkpoint per distinct target offset.")
    if any(symbol not in p.symbols for p in predictors):
        raise ValueError(f"Stock {symbol} must be in every checkpoint's stock universe.")


def actual_closes(predictor, symbol, indices):
    close = predictor.ohlcv[predictor.symbols.index(symbol), indices, 3]
    return np.where(np.isfinite(close) & (close > 0), close, np.nan)


def compare_independent_path(predictors, symbol, as_of):
    """Join several single-output forecasts into one path from the same signal."""
    primary = predictors[0]
    final_offset = max(p.target_offset for p in predictors)
    days = np.arange(1, final_offset + 1)
    targets = as_of + days
    known = targets < len(primary.dates)
    target_dates = np.full(final_offset, "", dtype="U10")
    target_dates[known] = primary.dates[targets[known]].astype(str)
    actual = np.full(final_offset, np.nan)
    actual[known] = actual_closes(primary, symbol, targets[known])
    predicted = np.full(final_offset, np.nan)

    for predictor in predictors:
        base, ratios = predictor.predict_path(as_of)
        stock = predictor.symbols.index(symbol)
        predicted[predictor.target_offset - 1] = base[stock] * ratios[stock, 0]

    if not np.isfinite(predicted).any():
        raise ValueError(f"No valid forecasts for {symbol} at the selected signal date.")
    return pd.DataFrame({
        "symbol": symbol,
        "signal_date": str(primary.dates[as_of]),
        "target_date": target_dates,
        "horizon": days,
        "actual_adjusted_close": actual,
        "predicted_adjusted_close": predicted,
        "error": predicted - actual,
    })


def compare_independent_range(predictors, symbol, start, end):
    """Keep the requested date range, with one aligned series per target offset."""
    primary = predictors[0]
    dates = primary.dates
    first = int(dates.searchsorted(start))
    stop = int(dates.searchsorted(end, side="right"))
    if first >= stop:
        raise ValueError("No trading sessions in the requested range.")

    actual = actual_closes(primary, symbol, slice(first, stop))
    result = pd.DataFrame({
        "symbol": symbol, "target_date": dates[first:stop].astype(str),
        "actual_adjusted_close": actual,
    })
    series = {}
    for predictor in sorted(predictors, key=lambda p: p.target_offset):
        lead = predictor.target_offset
        stock = predictor.symbols.index(symbol)
        signal_dates = np.full(stop - first, "", dtype="U10")
        predicted = np.full(stop - first, np.nan)
        earliest_signal = max(first, predictor.lookback - 1)
        for signal in range(earliest_signal, stop - lead):
            base, ratios = predictor.predict_path(signal)
            row = signal + lead - first
            predicted[row] = base[stock] * ratios[stock, 0]
            signal_dates[row] = str(dates[signal])
        result[f"signal_date_t_plus_{lead}"] = signal_dates
        column = f"predicted_t_plus_{lead}"
        result[column] = predicted
        result[f"error_t_plus_{lead}"] = predicted - actual
        series[column] = f"t+{lead} forecast"
    return result, series


def save_chart(result, output, title, series=None):
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
    if series is None:
        ax.plot(x, result["predicted_adjusted_close"], label="Predicted close",
                color="#bc4e21", linewidth=1.6, linestyle="--", marker=marker)
    else:
        for column, label in series.items():
            ax.plot(x, result[column], label=label, linewidth=1.6, linestyle="--", marker=marker)
    xlabel = "Trading sessions after signal date" if monthly else "Target trading date"
    ax.set(title=title, xlabel=xlabel, ylabel="Adjusted closing price (TWD)")
    if monthly:
        last = int(x.iloc[-1])
        ax.set_xticks(np.linspace(1, last, num=min(6, last), dtype=int))
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
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path)
    checkpoints.add_argument("--checkpoints", type=Path, nargs="+", help="One-output checkpoints for distinct target offsets.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--symbol", required=True, help="Stock code in the dataset, e.g. 2330.")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Use one completed signal date for single-output checkpoints.")
    parser.add_argument("--month", help="Fixed-origin forecast from the start of YYYY-MM using the checkpoint horizon.")
    parser.add_argument("--start", type=date.fromisoformat, help="First date, inclusive. With independent models this is also the earliest signal.")
    parser.add_argument("--end", type=date.fromisoformat, help="Last target date, inclusive. Defaults to the last data date.")
    parser.add_argument("--output", type=Path, default=Path("close_comparison.png"), help="New PNG path; CSV uses the same stem.")
    args = parser.parse_args()

    try:
        paths = args.checkpoints or [args.checkpoint]
        predictors = [StockPredictor(path, args.dataset) for path in paths]
        predictor = predictors[0]
        independent = args.checkpoints is not None or args.as_of is not None or predictor.target_offset != 1
        if independent:
            check_independent_models(predictors, args.symbol)
        test_dates = [p.first_test_date for p in predictors if p.first_test_date]
        test_start = max(test_dates, default=None)
        if args.as_of and (args.month or args.start or args.end):
            raise ValueError("--as-of cannot be combined with --month, --start or --end.")
        series = None
        if args.output.suffix.lower() != ".png":
            raise ValueError("--output must be a .png path.")
        csv_path = args.output.with_suffix(".csv")
        if args.output.exists() or csv_path.exists():
            raise ValueError("Output PNG or CSV already exists. Choose a new filename.")

        if args.as_of:
            signal_date = np.datetime64(args.as_of)
            as_of = int(predictor.dates.searchsorted(signal_date))
            if as_of >= len(predictor.dates) or predictor.dates[as_of] != signal_date:
                raise ValueError("--as-of must be a trading date present in the dataset.")
            result = compare_independent_path(predictors, args.symbol, as_of)
            title = f"{args.symbol} | Independent forecasts from {signal_date}"
        elif args.month:
            if args.start or args.end:
                raise ValueError("--month cannot be combined with --start or --end.")
            month_start = date.fromisoformat(args.month + "-01")
            if args.month != month_start.strftime("%Y-%m"):
                raise ValueError("--month must use YYYY-MM.")
            month = np.datetime64(month_start)
            if independent:
                as_of = month_signal_index(predictor, month)
                result = compare_independent_path(predictors, args.symbol, as_of)
            else:
                result = compare_month(predictor, args.symbol, month)
            signal_date = result["signal_date"].iat[0]
            title = f"{args.symbol} | {len(result)}-session forecast from {signal_date}"
            print(f"Month anchor: {args.month}-01 | last input session: {signal_date}")
        else:
            if not independent and predictor.horizon != 1:
                raise ValueError("Use --month YYYY-MM for a multi-session checkpoint.")
            start = args.start or test_start
            if start is None:
                raise ValueError("Older checkpoint: set --start to 'test starts' from the training log for test-only comparison.")
            start = np.datetime64(start, "D")
            end = np.datetime64(args.end, "D") if args.end else predictor.dates[-1]
            if start > end:
                raise ValueError("--start must be on or before --end.")
            if independent:
                result, series = compare_independent_range(predictors, args.symbol, start, end)
            else:
                result = compare_closes(predictor, args.symbol, start, end)
            title = f"{args.symbol} | Actual vs predicted close"

        if series is None:
            errors = result["error"].dropna()
            if not errors.empty:
                mae = errors.abs().mean()
                rmse = np.sqrt(errors.pow(2).mean())
                title += f"\nMAE {mae:.2f} TWD | RMSE {rmse:.2f} TWD"
                print(f"Paired sessions: {len(errors)}/{len(result)} | MAE: {mae:.4f} TWD | RMSE: {rmse:.4f} TWD")
        else:
            for column, label in series.items():
                errors = (result[column] - result["actual_adjusted_close"]).dropna()
                if errors.empty:
                    print(f"{label}: no paired sessions")
                else:
                    mae = errors.abs().mean()
                    rmse = np.sqrt(errors.pow(2).mean())
                    print(f"{label}: {len(errors)} pairs | MAE: {mae:.4f} TWD | RMSE: {rmse:.4f} TWD")
        known_dates = result.loc[result["target_date"].ne(""), "target_date"]
        if not known_dates.empty:
            print(f"Known target dates: {known_dates.iat[0]} to {known_dates.iat[-1]}")
            if test_start and known_dates.iat[0] < test_start:
                print(f"Includes dates before the shared checkpoint test period ({test_start}).")
        if "horizon" in result and len(known_dates) < len(result):
            print("Future dates absent from the dataset are shown by session number only.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_chart(result, args.output, title, series)
        result.to_csv(csv_path, index=False, mode="x")
        print(f"Saved {args.output} and {csv_path}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
