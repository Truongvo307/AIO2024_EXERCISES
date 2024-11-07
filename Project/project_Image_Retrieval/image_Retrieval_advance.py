
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from image_retrieval import read_image_from_path, folder_to_images, absolute_difference, mean_square_difference, cosine_similarity, correlation_coefficient
embedding_function = OpenCLIPEmbeddingFunction()
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = absolute_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = mean_square_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = cosine_similarity(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = correlation_coefficient(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# Optimizing the image retrieval process using the CLIP model


def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = f'{path}/{label}'
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = f'{label_path}/{filename}'
            files_path.append(filepath)
    return files_path


def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    collection.add(embeddings=embeddings, ids=ids)


def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(query_embeddings=[
                               query_embedding], n_results=n_results)  # how many results to return
    return results


if __name__ == '__main__':
    data_path = f'{ROOT}/train'
    files_path = get_files_path(path=data_path)
    # Create a Chroma Client
    chroma_client = chromadb.Client()
    # Create a collection
    l2_collection = chroma_client.get_or_create_collection(
        name="l2_collection", metadata={HNSW_SPACE: "l2"})
    add_embedding(collection=l2_collection, files_path=files_path)

    test_path = f'{ROOT}/test'
    test_files_path = get_files_path(path=test_path)
    test_path = test_files_path[1]
    l2_results = search(image_path=test_path,
                        collection=l2_collection, n_results=5)
    plot_results(image_path=test_path,
                 files_path=files_path, results=l2_results)
    # Create a collection
    cosine_collection = chroma_client.get_or_create_collection(name="Cosine_collection",
                                                               metadata={HNSW_SPACE: "cosine"})
    add_embedding(collection=cosine_collection, files_path=files_path)
