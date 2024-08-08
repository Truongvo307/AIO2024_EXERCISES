import numpy as np
import pandas as pd


class basic_probability:
    def __init__(self):
        pass

    def compute_mean(self, data):
        sum = 0
        for i in data:
            sum += i
        mean = sum / len(data)
        # mean = np.mean(data)
        return mean

    def compute_median(self, data):
        size = len(data)
        data.sort()
        if (size % 2 == 0):
            median = (data[size//2] + data[size//2 - 1]) / 2
        else:
            median = data[size//2]
        # np.median(data)
        return median

    def compute_std(self, data):
        mean = self.compute_mean(data)
        variance = 0
        for i in data:
            variance += (i - mean)**2
        std = np.sqrt(variance / len(data))
        return std

    def compute_correlation(self, data1, data2):
        N = len(data1)
        numberator = 0
        denominator = 0
        mean1 = self.compute_mean(data1)
        mean2 = self.compute_mean(data2)
        for i in data1:
            numerator = sum((data1[i] - mean1) * (data2[i] - mean2)
                            for i in range(N))
            denominator = np.sqrt(sum((data1[i] - mean1)**2 for i in range(N))) * np.sqrt(
                sum((data2[i] - mean2)**2 for i in range(N)))
        correlation = np.round(numerator / denominator, 2)
        return correlation


if __name__ == "__main__":
    data = np.array([2, 0, 2, 2, 7, 4, -2, 5, -1, -1])
    basic_probability = basic_probability()
    print(f'Mean : {basic_probability.compute_mean(data)}')
    X = [1, 5, 4, 4, 9, 13]
    print(f'Median : {basic_probability.compute_median(X)}')
