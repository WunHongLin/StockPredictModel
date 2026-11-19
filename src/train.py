import argparse
from pathlib import Path

import numpy as np
import torch

from download_tw50 import DEFAULT_DATASET, FEATURES
from evaluator import evaluate
from model import get_loss, StockMixer


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
args = parser.parse_args()
if args.epochs < 1:
    parser.error("--epochs must be positive")
if not args.dataset.is_file():
    parser.error("Dataset missing. Run python src/download_tw50.py first.")

with np.load(args.dataset, allow_pickle=False) as dataset:
    ohlcv = dataset["ohlcv"]
    dates = dataset["dates"]
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

model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
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
    window = ohlcv[:, offset:end].copy()
    base = np.where(mask_data[:, end - 1], ohlcv[:, end - 1, 3], 1.0)
    # Scale prices by the last observed close to compare stocks of different prices.
    window[:, :, :4] /= base[:, None, None]
    window[:, :, 4] = np.log1p(window[:, :, 4])
    window[~mask_data[:, offset:end]] = 0.0
    valid = mask_data[:, offset:target + 1].all(axis=1)
    returns = np.where(valid, ohlcv[:, target, 3] / base - 1, 0.0)
    return window, valid[:, None], np.ones_like(returns[:, None]), returns[:, None]


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

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')
