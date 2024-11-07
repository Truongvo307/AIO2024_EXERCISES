from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    df = pd.read_csv('cleveland.csv', header=None)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']

    # Điều chỉnh giá trị cột 'target' để gom các giá trị 2, 3, 4 thành 1 (người bị bệnh tim)
    df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    X = df.iloc[:, : -1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    classsifier = DecisionTreeClassifier(
        criterion='gini', max_depth=10, min_samples_split=2, random_state=42)
    classsifier.fit(X_train, y_train)
    y_pred = classsifier.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred)

    y_pred_train = classsifier.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)
    print(f'Accuracy for training set: {accuracy_for_train}')
    print(f'Accuracy for test set: {accuracy_for_test}')
