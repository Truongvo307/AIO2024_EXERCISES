import numpy as np


def compute_vector_length(v):
    len_of_vector = np.sqrt(np.sum(v**2))
    return len_of_vector


def compute_vector_dot_product(v1, v2):
    dot_product = np.dot(v1, v2)
    return dot_product


def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result


def matrix_multi_matrix(matrix1, matrix2):
    result = np.zeros((len(matrix1), len(matrix2[0])))
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def inverse_matrix(matrix):
    result = np.linalg.pinv(matrix)
    return result


def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


def compute_cosine_similarity(v1, v2):
    cosine_similarity = np.dot(
        v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosine_similarity


if __name__ == '__main__':
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    # Check if the matrix is square
    print(f"Length of vector: {compute_vector_length(v1)}")
    print(f"Dot product of two vectors: {compute_vector_dot_product(v1, v2)}")
    print(f"Matrix multi vector: {matrix_multi_vector(matrix1, v1)}")
    print(f"Matrix multi matrix: {matrix_multi_matrix(matrix1, matrix2)}")
    print(f"Inverse matrix: {inverse_matrix(matrix1)}")
    print(
        f"Eigenvalues and eigenvectors: {compute_eigenvalues_eigenvectors(matrix1)}")
    print(f"Cosine similarity: {compute_cosine_similarity(v1, v2)}")
