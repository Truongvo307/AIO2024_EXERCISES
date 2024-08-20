from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


DATASET_PATH = '2cls_spam_text_cls.csv'
# rehandle the messages


def lowercase(text):
    return text.lower()


def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)


def tokenize(text):

    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in tokens if token not in stop_words]


def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return tokens


# Create dict for round-truth data
def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


def predict(text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_features(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    print(df)
    messages = df['Message'].values.tolist()
    # print('messages:', messages)
    labels = df['Category'].values.tolist()
    # print('labels:', labels)
    # handle the labels
    le = LabelEncoder()  # unit lable [spam, ham] -> [0,1]
    y = le.fit_transform(labels)    # [0,1,...]
    # print(f'Classes: {le.classes_}')
    # print(f'Encoded labels: {y}')
    # handle the messages
    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)
    X = np.array([create_features(tokens, dictionary) for tokens in messages])

    # split the data
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    SEED = 0
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=VAL_SIZE,
                                                      shuffle=True,
                                                      random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        test_size=TEST_SIZE,
                                                        shuffle=True,
                                                        random_state=SEED)

    # Train the model
    model = GaussianNB()
    print('Start training...')
    model = model.fit(X_train, y_train)
    print('Training completed!')

    # Evaluate the model
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Val accuracy: {val_accuracy}')
    print(f'Test accuracy: {test_accuracy}')

    # Predict new messages
    # test_input = 'I am actually thinking a way of doing something useful'
    test_input = 'To use your credit'
    prediction_cls = predict(test_input, model, dictionary)
    print(f'Prediction: This is the {prediction_cls} email')
