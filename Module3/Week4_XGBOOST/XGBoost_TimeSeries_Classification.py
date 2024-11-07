import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


if __name__ == "__main__":
    x_train, y_train = readucr("FordA_TRAIN.tsv")
    x_test, y_test = readucr("FordA_TEST.tsv")

    classes = np . unique(np . concatenate((y_train, y_test), axis=0))
    plt.figure()
    for c in classes:
        c_x_train = x_train[y_train == c]
        plt.plot(c_x_train[0], label=" class " + str(c))

    plt.legend(loc="best")
    plt.show()
    plt.close()

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    model = XGBClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(" Confusion Matrix :")
    print(confusion_matrix(y_test, y_pred))
    print(" Classification Report :")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Accuracy : { accuracy :.2f}")
