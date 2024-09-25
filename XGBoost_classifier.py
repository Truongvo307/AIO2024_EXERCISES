import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    dataset_path = 'Problem4.csv'
    data_df = pd.read_csv(dataset_path)
    print(data_df)
    print(data_df.shape)
    print(data_df.info())

    # check categorical data
    categorical_cols = data_df.select_dtypes(
        include=['object', 'bool']).columns.to_list()
    print(categorical_cols)
    for col_name in categorical_cols:
        n_categories = data_df[col_name].nunique()
        print(f'Number of categories in {col_name}: {n_categories}')

    # Train test split
    X, y = data_df.iloc[:, :-1], data_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7)
    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of val samples: {X_test.shape[0]}')

    # Train model
    xg_class = xgb.XGBClassifier(seed=7)
    xg_class.fit(X_train, y_train)

    # Evaluation model
    preds = xg_class.predict(X_test)
    train_acc = accuracy_score(y_train, xg_class.predict(X_train))
    test_acc = accuracy_score(y_test, preds)

    print(f'Train ACC: {train_acc}')
    print(f'Test ACC: {test_acc}')

    # Plotting figure
    plt.figure()
    plt.scatter(X_test['alcohol'], y_test, s=20,
                edgecolor="black", c="green", label="True")
    plt.scatter(X_test['alcohol'], preds, s=20,
                edgecolor="black", c="darkorange", label="Prediction")
    plt.xlabel("alcohol")
    plt.ylabel("Target")
    plt.title("XGBoost Classification")
    plt.legend()
    plt.show()
