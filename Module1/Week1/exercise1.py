import numpy as np
from common_function import sigmoid, elu, relu, is_number
from loss_function import mae, mse, rmse


def input_compute_loss():
    num_samples = input("Input num_samples = ")
    if not num_samples.isnumeric():
        print("Num_samples must be a number")
        return
    func = input("Input loss name: ")
    num_samples = int(num_samples)
    rng = np.random.default_rng(30)
    for sample in range(num_samples):
        pred = rng.uniform(0, 10)
        target = rng.uniform(0, 10)(0, 10)
        if func == "MAE":
            loss = mae(pred, target, num_samples)
        elif func == "MSE":
            loss = mse(pred, target, num_samples)
        elif func == "RMSE":
            loss = rmse(pred, target, num_samples)
        print(
            f"loss name: {func}, sample: {sample}, pred: {pred}, target: {target}, loss:{loss}")


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
