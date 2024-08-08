import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
import seaborn as sns
from Week4.basic_probability import basic_probability


def correlation(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    denominator = np.sqrt(sum((x[i] - x_mean)**2 for i in range(len(x)))
                          * sum((y[i] - y_mean)**2 for i in range(len(y))))
    correlation = numerator / denominator
    return correlation


if __name__ == "__main__":
    data = pd.read_csv('advertising.csv')
    x = data['Radio']
    y = data['Sales']
    print(np.corrcoef(x, y))
    features = ['TV', 'Radio', 'Newspaper']
    # correlation between features
    for feature_1 in features:
        for feature_2 in features:
            correlation_value = correlation(data[feature_1], data[feature_2])
            print(
                f" Correlation between { feature_1 } and {feature_2}: { round(correlation_value , 2)}")

    data_cor = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_cor, annot=True)
    plt.show()
