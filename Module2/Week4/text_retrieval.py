import pandas as pd
import numpy as np
from sklearn . metrics . pairwise import cosine_similarity
from sklearn . feature_extraction . text import TfidfVectorizer


def tfidf_search(question, context_embedded, tfidf_vectorizer, top_d=5):
    # Lowercasing before encoding
    query = question.lower()

    # Transform the query into TF-IDF embedding
    query_embedded = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between the query and the context embeddings
    cosine_scores = cosine_similarity(
        query_embedded, context_embedded).flatten()

    # Get top k cosine scores and their indices
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)

    return results


def corr_search(question, context_embedded, tfidf_vectorizer, top_d=5):
    # Lowercasing before encoding
    query = question.lower()

    # Transform the query into TF-IDF embedding
    query_embedded = tfidf_vectorizer.transform([query])

    # Compute correlation between the query and the context embeddings
    corr_scores = cosine_similarity(query_embedded, context_embedded).flatten()

    # Get top k correlation scores and their indices
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)

    return results


if __name__ == "__main__":
    # Load the dataset
    vi_data_df = pd.read_csv("/Week4/vi_text_retrieval.csv")
    # Extract the text column
    context = vi_data_df['text']
    # Convert all text to lowercase
    context = [doc.lower() for doc in context]
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the context to get TF-IDF embeddings
    context_embedded = tfidf_vectorizer.fit_transform(context)
    # Convert the sparse matrix to a dense format and display the value at [7][0]
    dense_context_embedded = context_embedded.toarray()
    # Print the value at index [7][0]
    print(dense_context_embedded[7][0])
    # Sample question
    question = vi_data_df.iloc[0]['question']
    # Ensure context_embedded is defined before calling the function
    results1 = tfidf_search(question, context_embedded,
                            tfidf_vectorizer, top_d=5)
    # Print the cosine score of the top result
    print(results1[0]['cosine_score'])
    results2 = corr_search(question, context_embedded,
                           tfidf_vectorizer, top_d=5)
    # Print the correlation score of the top result
    print(results2[0]['corr_score'])
