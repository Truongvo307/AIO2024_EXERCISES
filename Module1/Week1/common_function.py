import numpy as np  # type: ignore


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








if __name__ == "__main__":
    print("Exercise 1 - Module 1 - 240601")
    print(f"1. Calculate F1 score: {cal_f1_score(2, 3, 4)}")
    print(f"2. Sigmoid: {sigmoid(2)}")
