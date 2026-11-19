"""Predict adjusted closing prices for the horizon stored in a StockMixer checkpoint."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from download_tw50 import DEFAULT_DATASET, FEATURES
from model import StockMixer


def prepare_window(window, valid_rows):
    """Apply the training normalization using only the input window."""
    window = window.copy()
    base_close = np.where(valid_rows[:, -1], window[:, -1, 3], 1.0)
    window[:, :, :4] /= base_close[:, None, None]
    window[:, :, 4] = np.log1p(window[:, :, 4])
    window[~valid_rows] = 0.0
    return window, base_close


class StockPredictor:
    def __init__(self, checkpoint_path, dataset_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]
        self.lookback = config["time_steps"]
        self.horizon = config.get("horizon", 1)
        self.first_test_date = checkpoint.get("first_test_date")

        with np.load(dataset_path, allow_pickle=False) as dataset:
            required = {"dates", "symbols", "features", "ohlcv"}
            if not required.issubset(dataset.files):
                raise ValueError("Dataset must contain dates, symbols, features and ohlcv.")
            self.dates = dataset["dates"].astype("datetime64[D]")
            symbols = dataset["symbols"].astype(str).tolist()
            features = dataset["features"].tolist()
            ohlcv = dataset["ohlcv"].astype(np.float32)

        if not checkpoint.get("symbols"):
            raise ValueError("Checkpoint lacks stock symbols. Retrain using a dataset with symbols.")
        self.symbols = [str(symbol) for symbol in checkpoint["symbols"]]
        if (
            len(self.symbols) != config["stocks"]
            or len(set(self.symbols)) != len(self.symbols)
            or len(set(symbols)) != len(symbols)
            or set(symbols) != set(self.symbols)
        ):
            raise ValueError("Dataset and checkpoint must contain the same distinct stocks.")
        if features != FEATURES or features != checkpoint.get("features") or config["channels"] != len(FEATURES):
            raise ValueError("Dataset feature order does not match the checkpoint.")
        if self.dates.ndim != 1 or np.isnat(self.dates).any() or np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError("Dataset dates must be valid, distinct and in ascending order.")
        if ohlcv.shape != (len(symbols), len(self.dates), len(FEATURES)):
            raise ValueError("Unexpected OHLCV dimensions.")
        if len(self.dates) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} trading sessions in the dataset.")

        positions = {symbol: index for index, symbol in enumerate(symbols)}
        self.ohlcv = ohlcv[[positions[symbol] for symbol in self.symbols]]
        self.valid_rows = np.isfinite(self.ohlcv).all(axis=2)
        self.model = StockMixer(**config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_path(self, as_of_index):
        """Return base closes and all forecast price ratios from one input window."""
        start = as_of_index - self.lookback + 1
        if start < 0 or as_of_index >= len(self.dates):
            raise ValueError(f"Prediction needs {self.lookback} completed trading sessions.")

        end = as_of_index + 1
        valid_rows = self.valid_rows[:, start:end]
        window, base_close = prepare_window(self.ohlcv[:, start:end], valid_rows)
        with torch.inference_mode():
            close_ratio = self.model(torch.from_numpy(window)).numpy()
        valid = valid_rows.all(axis=1, keepdims=True) & np.isfinite(close_ratio)
        return base_close, np.where(valid & (close_ratio > 0), close_ratio, np.nan)

    def predict(self, as_of_index):
        """Return next-session forecasts for a one-day checkpoint."""
        if self.horizon != 1:
            raise ValueError("This operation requires a one-session checkpoint. Use predict.py for multi-session forecasts.")
        base_close, ratios = self.predict_path(as_of_index)
        close_ratio = ratios[:, 0]

        valid = np.isfinite(close_ratio)
        result = pd.DataFrame({
            "symbol": self.symbols,
            "last_adjusted_close": base_close,
            "predicted_adjusted_close": base_close * close_ratio,
            "predicted_return": close_ratio - 1,
        })
        return result.loc[valid].sort_values(
            ["predicted_return", "symbol"], ascending=[False, True], ignore_index=True
        )


def forecast_table(predictor, as_of_index):
    dates = predictor.dates
    if predictor.horizon == 1:
        forecasts = predictor.predict(as_of_index)
        target = as_of_index + 1
        forecasts.insert(0, "target_date", str(dates[target]) if target < len(dates) else "")
    else:
        base, ratios = predictor.predict_path(as_of_index)
        stocks, horizons = np.where(np.isfinite(ratios))
        known = dates[as_of_index + 1:as_of_index + 1 + predictor.horizon]
        target_dates = np.full(predictor.horizon, "", dtype="U10")
        target_dates[:len(known)] = known.astype(str)
        forecasts = pd.DataFrame({
            "target_date": target_dates[horizons],
            "horizon": horizons + 1,
            "symbol": np.asarray(predictor.symbols)[stocks],
            "last_adjusted_close": base[stocks],
            "predicted_adjusted_close": base[stocks] * ratios[stocks, horizons],
            "predicted_return": ratios[stocks, horizons] - 1,
        }).sort_values(
            ["horizon", "predicted_return", "symbol"],
            ascending=[True, False, True], ignore_index=True,
        )
    forecasts.insert(0, "as_of", str(dates[as_of_index]))
    return forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--as-of", type=date.fromisoformat, help="Completed trading date; defaults to the last row.")
    parser.add_argument("--output", type=Path, help="Optional path for a new forecast CSV.")
    args = parser.parse_args()

    try:
        predictor = StockPredictor(args.checkpoint, args.dataset)
        as_of = np.datetime64(args.as_of) if args.as_of else predictor.dates[-1]
        index = int(predictor.dates.searchsorted(as_of))
        if index == len(predictor.dates) or predictor.dates[index] != as_of:
            raise ValueError("--as-of must be a trading date present in the dataset.")
        forecasts = forecast_table(predictor, index)
        if forecasts.empty:
            raise ValueError("No valid stock forecasts on the requested date.")

        if args.output:
            forecasts.to_csv(args.output, index=False, mode="x")
        if predictor.horizon == 1:
            target = forecasts["target_date"].iat[0] or "next trading session"
            print(f"Signal date: {as_of}; target: {target}")
        else:
            print(f"Signal date: {as_of}; horizon: {predictor.horizon} trading sessions")
        print(forecasts.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
