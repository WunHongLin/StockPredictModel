import numpy as np
import pandas as pd


def evaluate(prediction, ground_truth, mask, report=False):
    """Evaluate daily cross-sectional predictions using valid stocks only."""
    if prediction.shape != ground_truth.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction, ground truth and mask shapes must match.")

    valid = mask >= 0.5
    if not valid.any():
        raise ValueError("No valid predictions to evaluate.")

    ic = []
    rank_ic = []
    precision = []
    top5_returns = []

    for day, day_mask in enumerate(valid.T):
        if not day_mask.any():
            continue
        scores = prediction[day_mask, day]
        returns = ground_truth[day_mask, day]
        if np.ptp(scores) > 0 and np.ptp(returns) > 0:
            ic.append(np.corrcoef(scores, returns)[0, 1])
            # Pearson correlation of average ranks gives Spearman correlation with ties.
            ranks = pd.DataFrame({"scores": scores, "returns": returns}).rank().to_numpy()
            rank_ic.append(np.corrcoef(ranks.T)[0, 1])

        ranked_returns = returns[np.argsort(scores, stable=True)[::-1]]
        precision.append(np.mean(ranked_returns[:10] >= 0))
        top5_returns.append(np.mean(ranked_returns[:5]))

    top5_volatility = np.std(top5_returns)
    return {
        "mse": np.mean((prediction[valid] - ground_truth[valid]) ** 2),
        "IC": np.mean(ic) if ic else np.nan,
        "RIC": np.mean(rank_ic) if rank_ic else np.nan,
        "prec_10": np.mean(precision),
        "sharpe5": np.sqrt(252) * np.mean(top5_returns) / top5_volatility
        if top5_volatility > 0 else np.nan,
    }
