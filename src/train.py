import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
import torch

from download_tw50 import DEFAULT_DATASET, FEATURES
from evaluator import evaluate
from model import get_loss, StockMixer
from predict import prepare_window


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
print(device)

market_num = 20
learning_rate = 0.001
alpha = 0.1
scale_factor = 3

parser = argparse.ArgumentParser(description="Train StockMixer on TWSE OHLCV data.")
parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lookback", type=int, default=32,
                    help="Number of past trading sessions used as input.")
parser.add_argument("--horizon", "--horizen", type=int, default=1,
                    help="Number of future trading sessions predicted in one pass.")
parser.add_argument("--target-offset", type=int, default=1,
                    help="Trading sessions from the input's last close to the first target.")
parser.add_argument(
    "--patience", type=int, default=10,
    help="Stop after this many epochs without lower validation loss. Use 0 to disable.",
)
parser.add_argument(
    "--checkpoint", type=Path,
    help="File for best model weights. Defaults to checkpoints/run-*/best_stockmixer.pt.",
)
args = parser.parse_args()
if args.epochs < 1:
    parser.error("--epochs must be positive")
if args.lookback < 2:
    parser.error("--lookback must be at least 2")
if args.horizon < 1:
    parser.error("--horizon must be positive")
if args.target_offset < 1:
    parser.error("--target-offset must be positive")
if args.target_offset != 1 and args.horizon != 1:
    parser.error("--target-offset greater than 1 requires --horizon 1")
if args.patience < 0:
    parser.error("--patience must be non-negative")
if args.checkpoint is not None and args.checkpoint.resolve() == args.dataset.resolve():
    parser.error("--checkpoint and --dataset must be different files")
if not args.dataset.is_file():
    parser.error("Dataset missing. Run python src/download_tw50.py first.")

with np.load(args.dataset, allow_pickle=False) as dataset:
    ohlcv = dataset["ohlcv"]
    dates = dataset["dates"]
    symbols = dataset["symbols"].tolist() if "symbols" in dataset else None
    if dataset["features"].tolist() != FEATURES:
        parser.error("Unexpected feature order. Regenerate the dataset.")
stock_num, trade_dates, fea_num = ohlcv.shape
lookback_length = args.lookback
steps = args.horizon
target_offset = args.target_offset
target_start_offset = lookback_length + target_offset - 1
valid_index = int(trade_dates * 0.7)
test_index = int(trade_dates * 0.85)
train_windows = valid_index - target_start_offset - steps + 1
required_target_span = target_offset + steps - 1
if train_windows < 1 or min(test_index - valid_index, trade_dates - test_index) < required_target_span:
    parser.error(
        f"Insufficient history for lookback={lookback_length}, horizon={steps}, target-offset={target_offset}. "
        f"Need at least {target_start_offset + steps} training sessions and "
        f"{required_target_span} sessions each in validation and testing."
    )
mask_data = np.isfinite(ohlcv).all(axis=2)
print(f"{stock_num} stocks, {dates[0]} to {dates[-1]}, "
      f"validation starts {dates[valid_index]}, test starts {dates[test_index]}")
print(f"Lookback: {lookback_length} | Outputs: {steps} | First target: t+{target_offset} trading sessions")

model_config = {
    "stocks": stock_num, "time_steps": lookback_length, "channels": fea_num,
    "market": market_num, "scale": scale_factor, "horizon": steps,
}
model = StockMixer(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def target_mask(offset):
    end = offset + lookback_length
    target_start = offset + target_start_offset
    return mask_data[:, offset:end].all(axis=1, keepdims=True) & mask_data[:, target_start:target_start + steps]


def offsets_for_split(start, end):
    # Start no earlier than the prior split's last close as the signal date.
    # Keep all targets within this split so the forecast never uses future labels.
    return [
        offset for offset in range(max(0, start - lookback_length), end - target_start_offset - steps + 1)
        if target_mask(offset).any()
    ]


batch_offsets = offsets_for_split(0, valid_index)
valid_offsets = offsets_for_split(valid_index, test_index)
test_offsets = offsets_for_split(test_index, trade_dates)
if not (batch_offsets and valid_offsets and test_offsets):
    parser.error("Each split needs at least one forecast window with valid data.")


def get_batch(offset):
    end = offset + lookback_length
    window, base = prepare_window(ohlcv[:, offset:end], mask_data[:, offset:end])
    valid = target_mask(offset)
    target_start = offset + target_start_offset
    target_closes = ohlcv[:, target_start:target_start + steps, 3]
    returns = np.where(valid, target_closes / base[:, None] - 1, 0.0)
    return window, valid, np.ones_like(base[:, None]), returns, base


def validate(offsets):
    model.eval()
    losses = np.zeros(3)
    predictions, targets, masks, base_closes = [], [], [], []
    with torch.no_grad():
        for offset in offsets:
            window, valid, price, returns, base = get_batch(offset)
            inputs, mask, price, target = (
                torch.as_tensor(x, device=device, dtype=torch.float32)
                for x in (window, valid, price, returns)
            )
            loss, reg, rank, forecast = get_loss(model(inputs), target, price, mask, stock_num, alpha)
            losses += [loss.item(), reg.item(), rank.item()]
            predictions.append(forecast.cpu().numpy())
            targets.append(returns)
            masks.append(valid)
            base_closes.append(base)
    predicted, actual, valid = (np.stack(rows) for rows in (predictions, targets, masks))
    if steps == 1 and target_offset == 1:
        metrics = evaluate(predicted[:, :, 0].T, actual[:, :, 0].T, valid[:, :, 0].T)
    else:
        errors = (predicted - actual) * np.stack(base_closes)[:, :, None]
        metrics = {}
        last_day = target_offset + steps - 1
        report_days = sorted({
            day for day in (1, 5, 10, 20) if target_offset <= day <= last_day
        } | {last_day})
        for day in report_days:
            index = day - target_offset
            paired = errors[:, :, index][valid[:, :, index]]
            metrics[day] = (
                float(np.abs(paired).mean()) if paired.size else np.nan,
                float(np.sqrt(np.square(paired).mean())) if paired.size else np.nan,
                paired.size,
            )
    return (*map(float, losses / len(offsets)), metrics)


def save_checkpoint(path, checkpoint):
    # Write a complete checkpoint before replacing the previous best.
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def print_performance(split, metrics):
    if steps > 1 or target_offset > 1:
        for day, (mae, rmse, samples) in metrics.items():
            print(f"{split} day +{day}: MAE={mae:.4f} TWD, RMSE={rmse:.4f} TWD, pairs={samples}")
            return
    print(
        f"{split} performance: mse:{metrics['mse']:.2e}, IC:{metrics['IC']:.2e}, "
        f"RIC:{metrics['RIC']:.2e}, prec@10:{metrics['prec_10']:.2e}, "
        f"SR:{metrics['sharpe5']:.2e}"
    )


checkpoint_path = args.checkpoint
if checkpoint_path is None:
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = Path(mkdtemp(prefix="run-", dir=checkpoint_dir)) / "best_stockmixer.pt"
else:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

best_valid_loss = np.inf
epochs_without_improvement = 0

for epoch in range(args.epochs):
    model.train()
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for offset in batch_offsets:
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(offset)[:4]
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / len(batch_offsets)
    tra_reg_loss = tra_reg_loss / len(batch_offsets)
    tra_rank_loss = tra_rank_loss / len(batch_offsets)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_offsets)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    print_performance("Valid", val_perf)
    if not np.isfinite(val_loss):
        raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}.")

    if val_loss < best_valid_loss:
        save_checkpoint(checkpoint_path, {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch + 1,
            "val_loss": float(val_loss),
            "symbols": symbols,
            "features": FEATURES,
            "first_test_date": str(dates[test_index + target_offset - 1]),
            "target_offset": target_offset,
        })
        best_valid_loss = val_loss
        epochs_without_improvement = 0
        print(f"Saved best checkpoint to {checkpoint_path}")
    else:
        epochs_without_improvement += 1
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={args.patience}).")
            break

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
print(
    f"Best epoch: {checkpoint['epoch']}; "
    f"validation loss: {checkpoint['val_loss']:.6e}; epochs run: {epoch + 1}"
)
test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_offsets)
print(f"Test: loss:{test_loss:.2e} = {test_reg_loss:.2e} + alpha*{test_rank_loss:.2e}")
print_performance("Test", test_perf)
