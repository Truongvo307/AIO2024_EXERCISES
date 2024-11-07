import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # DATA
    dataset_path = 'Problem3.csv'
    data_df = pd.read_csv(dataset_path)
    # print(data_df)
    # print(data_df.shape)
    # print(data_df.info())

    # Load dataset
    categorical_cols = data_df.select_dtypes(
        include=['object', 'bool']).columns.to_list()
    print(categorical_cols)
    for col_name in categorical_cols:
        n_categories = data_df[col_name].nunique()
        print(f'Number of categories in {col_name}: {n_categories}')

    # Encode categorical data
    ordinal_encoder = OrdinalEncoder()
    encoded_categorical_cols = ordinal_encoder.fit_transform(
        data_df[categorical_cols])

    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_cols, columns=categorical_cols)
    numerical_df = data_df.drop(categorical_cols, axis=1)
    encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)

    print(encoded_df)

    # Split data
    X = encoded_df.drop(columns=['area'])
    y = encoded_df['area']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7)
    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of val samples: {X_test.shape[0]}')

    # Train model
    xg_reg = xgb.XGBRegressor(
        seed=7, learning_rate=0.01, n_estimators=102, max_depth=3)
    xg_reg.fit(X_train, y_train)

    # Evaluate model
    preds = xg_reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print('Evaluation results on test set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    # Plotting
    plt.figure()
    plt.scatter(X_test['temp'], y_test, s=20,
                edgecolor="black", c="green", label="True")
    plt.scatter(X_test['temp'], preds, s=20, edgecolor="black",
                c="darkorange", label="Prediction")
    plt.xlabel("temp")
    plt.ylabel("area")
    plt.title("XGBoost Regression")
    plt.legend()
    plt.show()
