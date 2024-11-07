from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self, data):
        np.random.seed(42)
        self.centroids = data[np.random.choice(
            data.shape[0], self.k, replace=False)]

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def assign_clusters(self, data):
        distances = np.array([[self.euclidean_distance(x, centroid) for centroid in self.centroids]
                              for x in data])
        return np.argmin(distances, axis=1)

    def update_centroids(self, data):
        self.centroids = np.array(
            [data[self.clusters == k].mean(axis=0) for k in range(self.k)])

    def plot_clusters(self, data, iter):
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters,
                    cmap='viridis', marker='o', alpha=0.6)
        plt.scatter(
            self.centroids[:, 0], self.centroids[:, 1], s=300, c='blue', marker='x')
        plt.title(f"Iteration {iter + 1}")
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        plt.show()

    def plot_final_clusters(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters,
                    cmap='viridis', marker='o', alpha=0.6)
        plt.scatter(
            self.centroids[:, 0], self.centroids[:, 1], s=300, c='blue', marker='x')
        plt.title("Final Clustering")
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.show()

    def fit(self, data):
        self.initialize_centroids(data)
        for i in range(self.max_iters):
            self.clusters = self.assign_clusters(data)
            self.plot_clusters(data, i)
            old_centroids = self.centroids.copy()
            self.update_centroids(data)
            if np.all(old_centroids == self.centroids):
                break


if __name__ == "__main__":
    # Load the iris dataset
    iris_dataset = load_iris()
    data = iris_dataset.data
    data = iris_dataset.data[:, :2]

    # Plot data
    plt.scatter(data[:, 0], data[:, 1], c='gray')
    plt.title("Initial Dataset")
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

    kmeans = KMeans(k=2, max_iters=100)
    kmeans.fit(data)
    kmeans = KMeans(k=3, max_iters=100)
    kmeans.fit(data)
    kmeans = KMeans(k=4, max_iters=100)
    kmeans.fit(data)
