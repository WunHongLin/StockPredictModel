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

lookback_length = 16
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3

parser = argparse.ArgumentParser(description="Train StockMixer on TWSE OHLCV data.")
parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
parser.add_argument("--epochs", type=int, default=100)
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
valid_index = int(trade_dates * 0.7)
test_index = int(trade_dates * 0.85)
train_windows = valid_index - lookback_length - steps + 1
if train_windows < 1 or not valid_index < test_index < trade_dates:
    parser.error("Insufficient history for training, validation and testing.")
mask_data = np.isfinite(ohlcv).all(axis=2)
print(f"{stock_num} stocks, {dates[0]} to {dates[-1]}, "
      f"validation starts {dates[valid_index]}, test starts {dates[test_index]}")

model_config = {
    "stocks": stock_num, "time_steps": lookback_length, "channels": fea_num,
    "market": market_num, "scale": scale_factor,
}
model = StockMixer(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def has_valid_stock(offset):
    return mask_data[:, offset:offset + lookback_length + steps].all(axis=1).any()


batch_offsets = np.array([offset for offset in range(train_windows) if has_valid_stock(offset)])
if not len(batch_offsets):
    parser.error("No valid training windows.")


def validate(start_index, end_index):
    model.eval()
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            if not has_valid_stock(cur_offset):
                continue
            data_batch, mask_batch, price_batch, gt_batch = map(

                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        active = cur_valid_mask.any(axis=0)
        if not active.any():
            raise ValueError("No valid evaluation windows. Download a longer date range.")
        count = active.sum()
        cur_valid_perf = evaluate(
            cur_valid_pred[:, active], cur_valid_gt[:, active], cur_valid_mask[:, active]
        )
    return loss / count, reg_loss / count, rank_loss / count, cur_valid_perf


def get_batch(offset):
    end = offset + lookback_length
    target = end + steps - 1
    window, base = prepare_window(ohlcv[:, offset:end], mask_data[:, offset:end])
    valid = mask_data[:, offset:target + 1].all(axis=1)
    returns = np.where(valid, ohlcv[:, target, 3] / base - 1, 0.0)
    return window, valid[:, None], np.ones_like(returns[:, None]), returns[:, None]


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
            get_batch(offset)
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

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
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
            "first_test_date": str(dates[test_index]),
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
test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
print(f"Test: loss:{test_loss:.2e} = {test_reg_loss:.2e} + alpha*{test_rank_loss:.2e}")
print_performance("Test", test_perf)
