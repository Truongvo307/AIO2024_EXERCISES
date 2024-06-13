import numpy as np  # type: ignore


def cal_f1_score(tp, fp, fn):
    failure = False
    if not isinstance(tp, int):
        failure = True
        print(f"tp must be int -- currently input {type(tp)}")
    if not isinstance(fp, int):
        failure = True
        print(f"fp must be int -- currently input {type(fp)}")
    if not isinstance(fn, int):
        failure = True
        print(f"fn must be int -- currently input {type(fn)}")
    if failure:
        return
    if tp*fp*fn == 0:
        print("tp and fb and fn must be greater than zero")
        return
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * (precision*recall)/(precision+recall)
    return f1_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def relu(x):
    return np.maximum(0, x)


def is_number(n):
    try:
        float(n)  # Type - casting the string to ‘float ‘.
    # If string is not a valid ‘float ‘ ,
    # it ’ll raise ‘ValueError ‘ exception
    except ValueError:
        return False
    return True


def input_compute_act():
    x = input("Input x = ")
    if not is_number(x):
        print("x must be a number")
        return
    func = input("Input activation function name (sigmoid|Relu|Elu): ")
    x = float(x)
    if func == "sigmoid":
        return print(f"{func}: f({x}) = {sigmoid(x)}")
    elif func == "relu":
        return print(f"{func}: f({x}) = {relu(x)}")
    elif func == " elu":
        return print(f"{func}: f({x}) = {elu(x)}")
    else:
        print(f"{func} is NOT supportted")


def mae(pred, target, num_samples):
    return (np.abs(pred-target)/num_samples)


def mse(pred, target, num_samples):
    return (((target - pred) ** 2)/num_samples)


def rmse(pred, target, num_samples):
    return np.sqrt(mse(target, pred, num_samples))


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


def approx_sin(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    sin_approx = 0
    for n in range(n):
        term = ((-1)**n) * (x**(2*n+1)) / factorial(2*n+1)
        sin_approx += term
    print(sin_approx)


def approx_cos(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    cos_approx = 0
    for n in range(n):
        term = ((-1)**n) * (x**(2*n)) / factorial(2*n)
        cos_approx += term
    print(cos_approx)


def approx_sinh(x, n):
    if not (0 <= x <= 2 * np.pi):
        print(f"{x}: x must be a radian value")
        return
    if (n <= 0):
        print(f"{n}: n must be greater than 0")
        return
    sinh_approx = 0
    for n in range(n):
        term = (x**(2*n+1)) / factorial(2*n+1)
        sinh_approx += term
    print(sinh_approx)


def approx_cosh(x, n):
    if not (0 <= x <= 2 * np.pi):
        print("x must be a radian value")
        return
    if (n <= 0):
        print("n must be greater than 0")
        return
    cosh_approx = 0
    for n in range(n):
        term = (x**(2*n)) / factorial(2*n)
        cosh_approx += term
    print(cosh_approx)


def mdre(y, y_hat, n, p):
    mdre = (y ** (1/n) - y_hat ** (1/n)) ** p
    print(mdre)


if __name__ == "__main__":
    print("Exercise 1 - Module 1 - 240601")
    print(f"1. Calculate F1 score: {cal_f1_score(2, 3, 4)}")
    print(f"2. Sigmoid: {sigmoid(2)}")
