
import numpy as np
import random
import matplotlib.pyplot as plt


def get_column(data, index):
    # Extract the specified column from the 2D list
    result = [row[index] for row in data]
    return result


def prepare_data(file_name_dataset):
    # Load the dataset from a CSV file
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()
    N = len(data)
    # Get TV data (index = 0)
    tv_data = get_column(data, 0)
    # Get radio data (index = 1)
    radio_data = get_column(data, 1)
    # Get newspaper data (index = 2)
    newspaper_data = get_column(data, 2)
    # Get sales data (index = 3)
    sales_data = get_column(data, 3)

    # Building X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return X, y


def initialize_params():
    # Initialize weights and bias with predefined values
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b = 0
    w1 = 0.016992259082509283  # Weight for feature 1
    w2 = 0.0070783670518262355  # Weight for feature 2
    w3 = -0.002307860847821344  # Weight for feature 3
    b = 0.0  # Bias term
    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    result = (w1 * x1) + (w2 * x2) + (w3 * x3) + b
    return result


def compute_loss_mae(y, y_hat):
    return abs(y - y_hat)


def compute_loss_mse(y, y_hat):
    # Mean Squared Error loss
    return (y - y_hat) ** 2


def compute_gradient_wi(xi, y, y_hat):
    # Gradient for weight wi
    return -2 * xi * (y - y_hat)


def compute_gradient_b(y, y_hat):
    # Gradient for bias b
    return -2 * (y - y_hat)


def update_weight_wi(wi, dl_dwi, lr):
    # Update weight wi
    return wi - lr * dl_dwi


def update_weight_b(b, dl_db, lr):
    # Update bias b
    return b - lr * dl_db


def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    # Initialize parameters
    w1, w2, w3, b = initialize_params()

    N = len(y_data)
    for epoch in range(epoch_max):
        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0
        for i in range(N):
            # Get a sample
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]

            # Compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # Compute loss
            loss_total += compute_loss_mse(y, y_hat)

            # Compute gradients
            dw1_total += compute_gradient_wi(x1, y, y_hat)
            dw2_total += compute_gradient_wi(x2, y, y_hat)
            dw3_total += compute_gradient_wi(x3, y, y_hat)
            db_total += compute_gradient_b(y, y_hat)

            # Update parameters
        w1 = update_weight_wi(w1, dw1_total/N, lr)
        w2 = update_weight_wi(w2, dw2_total/N, lr)
        w3 = update_weight_wi(w3, dw3_total/N, lr)
        b = update_weight_b(b, db_total/N, lr)

        # Log the loss at the end of each epoch
        losses.append(loss_total/N)
    return w1, w2, w3, b, losses


if __name__ == '__main__':
    X, y = prepare_data('advertising.csv')
    (w1, w2, w3, b, losses) = implement_linear_regression(
        X, y, epoch_max=1000, lr=1e-5)
    # print(losses)
    print(w1, w2, w3)
    plt.plot(losses)
    plt.xlabel("# epoch ")
    plt.ylabel("MSE Loss ")
    plt.show()
