import numpy as np
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def IA(pred, true):
    mean_true = np.mean(true)
    return 1 - (np.sum((true - pred) ** 2)) / (
        np.sum((np.abs(pred - mean_true) + np.abs(true - mean_true)) ** 2))


def Pearson(pred, true):
    return pearsonr(pred, true)

def kde_relative_residual_interval(pred, true, ci=95, grid_size=5000, bw_method="scott"):
    """
    KDE-based empirical prediction interval using relative residuals.
    Recommended when the target scale varies with depth.
    """
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    if len(pred) != len(true):
        raise ValueError("pred and true must have the same length.")

    eps = 1e-8
    rel_residual = (true - pred) / (pred + eps)

    kde = gaussian_kde(rel_residual, bw_method=bw_method)

    std = np.std(rel_residual)
    x_min = rel_residual.min() - 3 * std
    x_max = rel_residual.max() + 3 * std
    grid = np.linspace(x_min, x_max, grid_size)

    pdf = kde(grid)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]

    alpha = (100 - ci) / 2 / 100
    q_low = np.interp(alpha, cdf, grid)
    q_high = np.interp(1 - alpha, cdf, grid)

    lower = pred * (1 + q_low)
    upper = pred * (1 + q_high)

    covered = (true >= lower) & (true <= upper)
    picp = np.mean(covered) * 100
    mpiw = np.mean(upper - lower)

    return {
        "PICP": picp,
        "MPIW": mpiw,
        "q_low": q_low,
        "q_high": q_high,
        "lower": lower,
        "upper": upper
    }

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    ia = IA(pred, true)
    R = Pearson(pred, true)
    result = kde_relative_residual_interval(pred, true, ci=95)

    return mae, mse, rmse, mape, mspe, ia, R, result['PICP'], result['MPIW']

