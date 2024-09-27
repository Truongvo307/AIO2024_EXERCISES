# Load dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english'))


def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Remove duplicate rows
        df = df.drop_duplicates()
        return df
    except Exception as e:
        print(e)
        return None


def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data


def preprocess_text(text):
    wl = WordNetLemmatizer()
    # Removing HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Expanding chatwords and contractions
    text = expand_contractions(text)
    # Removing emojis
    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'', text)
    # Adding space after full stops
    text = re.sub(r'\.(?=\S)', '. ', text)
    # Removing URLs
    text = re.sub(r'http\S+', '', text)
    # Removing punctuation and converting to lowercase
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    # Lemmatizing and removing stop words
    text = " ".join([wl.lemmatize(word) for word in text.split()
                    if word not in stop and word.isalpha()])

    return text

# Creating autocpt arguments


def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# analyzing the sentiment distribution in the dataset


def sentiment_distribution(df):
    # Calculating frequencies of positive and negative sentiments
    freq_pos = len(df[df['sentiment'] == 'positive'])
    freq_neg = len(df[df['sentiment'] == 'negative'])

    # Data for pie chart
    data = [freq_pos, freq_neg]
    labels = ['positive', 'negative']

    # Create pie chart
    fig, ax = plt.subplots(figsize=[11, 7])
    plt.pie(x=data, autopct=lambda pct: func(pct, data), explode=[0.025]*2,
            pctdistance=0.5, colors=[sns.color_palette()[0], 'tab:red'],
            textprops={'fontsize': 16})
    # Add legend and title
    plt.legend(labels, loc="best", prop={'size': 14})
    plt.title('Frequencies of Sentiment Labels',
              fontsize=14, fontweight='bold')
    # Save pie chart
    fig.savefig("PieChart.png")
    plt.show()

    # Length of data samples


def samples_length_analysis(df):
    words_len = df['review'].str.split().map(lambda x: len(x))
    df_temp = df.copy()
    df_temp['words length'] = words_len

    # Histogram for positive reviews
    hist_positive = sns.displot(
        data=df_temp[df_temp['sentiment'] == 'positive'],
        x='words length', hue='sentiment', kde=True, height=7, aspect=1.1, legend=False
    ).set(title='Words in Positive Reviews')
    plt.show()

    # Histogram for negative reviews
    hist_negative = sns.displot(
        data=df_temp[df_temp['sentiment'] == 'negative'],
        x='words length', hue='sentiment', kde=True, height=7, aspect=1.1, legend=False, palette=['red']
    ).set(title='Words in Negative Reviews')
    plt.show()

    # Kernel Density Plot for words in reviews
    plt.figure(figsize=(7, 7.1))
    kernel_distribution_number_words_plot = sns.kdeplot(
        data=df_temp, x='words length', hue='sentiment', fill=True, palette=[sns.color_palette()[0], 'red']
    ).set(title='Words in Reviews')
    plt.legend(title='Sentiment', labels=['negative', 'positive'])
    plt.show()


if __name__ == "__main__":
    file_path = './IMDB-Dataset.csv'
    df = read_data(file_path)
    print(df.head())
    # clean data
    df['review'] = df['review'].apply(preprocess_text)
    # Encoding the labels
    label_encode = LabelEncoder()
    y_data = label_encode.fit_transform(df['sentiment'])

    # Data Analysis
    sentiment_distribution(df)
    samples_length_analysis(df)

    print('--------------Slpit data & Handle dataset----------------')
    # Splitting the dataset
    x_data = df['review']  # Assuming 'review' column contains the text data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    # Text to vector
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_vectorizer .fit(x_train, y_train)
    x_train_encoded = tfidf_vectorizer.transform(x_train)
    x_test_encoded = tfidf_vectorizer.transform(x_test)

    print('--------------Training and Evaluation----------------')
    # Training and Evaluating the mode
    dt_classifier = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42
    )
    dt_classifier.fit(x_train_encoded, y_train)
    y_pred_dt = dt_classifier.predict(x_test_encoded)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

    # Training and evaluating the Random Forest model
    rf_classifier = RandomForestClassifier(
        random_state=42
    )
    rf_classifier.fit(x_train_encoded, y_train)
    y_pred_rf = rf_classifier.predict(x_test_encoded)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
