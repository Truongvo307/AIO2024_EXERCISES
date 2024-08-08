import numpy as np


def mae(pred, target, num_samples):
    return (np.abs(pred-target)/num_samples)


def mse(pred, target, num_samples):
    return (((target - pred) ** 2)/num_samples)


def rmse(pred, target, num_samples):
    return np.sqrt(mse(target, pred, num_samples))


def mdre(y, y_hat, n, p):
    mdre = (y ** (1/n) - y_hat ** (1/n)) ** p
    return mdre
