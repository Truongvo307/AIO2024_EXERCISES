import numpy as np
import random
import matplotlib.pyplot as plt

# Function to prepare data


def prepare_data(file_name_dataset):
    # Load dataset and skip the header row
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()

    # Get columns for TV, Radio, Newspaper, and Sales
    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    # Create input X with a bias term and features, and output y
    X = [[1, x1, x2, x3]
         for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y

# Helper function to extract column data


def get_column(data, index):
    return [row[index] for row in data]

# Function to initialize weights


def initialize_params():
    # Randomly initialize weights and bias
    bias = 0
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)

    # Return initialized weights (or fixed values for testing)
    return [0, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845]

# Predict output using the linear equation y = x0*b + x1*w1 + x2*w2 + x3*w3


def predict(X_features, weights):
    return sum([x * w for x, w in zip(X_features, weights)])

# Compute loss (Mean Squared Error)


def compute_loss(y_hat, y):
    return (y_hat - y) ** 2

# Compute gradients for the weights based on the loss


def compute_gradient_w(X_features, y, y_hat):
    error = y_hat - y
    # Gradient of loss w.r.t each weight
    dl_dweights = [error * x for x in X_features]
    return dl_dweights

# Update the weights using the computed gradients and learning rate


def update_weight(weights, dl_dweights, lr):
    return [w - lr * dw for w, dw in zip(weights, dl_dweights)]

# Function to implement linear regression training loop


def implement_linear_regression(X_feature, y_output, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()  # Initialize weights
    N = len(y_output)  # Number of samples

    for epoch in range(epoch_max):
        print("Epoch:", epoch)
        for i in range(N):
            # Get a sample (row i)
            features_i = X_feature[i]
            y = y_output[i]

            # Compute predicted output
            y_hat = predict(features_i, weights)

            # Compute loss
            loss = compute_loss(y_hat, y)

            # Compute gradients for weights
            dl_dweights = compute_gradient_w(features_i, y, y_hat)

            # Update weights
            weights = update_weight(weights, dl_dweights, lr)

            # Logging loss
            losses.append(loss)
    print('break')
    return weights, losses


# Main workflow
X, y = prepare_data('advertising.csv')  # Load the data
W, L = implement_linear_regression(X, y, epoch_max=50, lr=1e-5)
print(L[9999])
# Plot the loss curve
# plt.plot(L[0:100])  # Plot first 100 loss values for visualization
# plt.xlabel("# iteration")
# plt.ylabel("Loss")
# plt.show()
