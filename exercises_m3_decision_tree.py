import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == '__main__':
    dataset_path = './Housing.csv'
    df = pd.read_csv(dataset_path)
    print(df.head())
    categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
    print(categorical_cols)
    # Assuming df is your DataFrame and categorical_cols is a list of categorical column names
    ordinal_encoder = OrdinalEncoder()
    encoded_categorical_cols = ordinal_encoder.fit_transform(
        df[categorical_cols])

    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_cols,
        columns=categorical_cols
    )

    numerical_df = df.drop(categorical_cols, axis=1)

    encoded_df = pd.concat(
        [numerical_df, encoded_categorical_df],
        axis=1
    )
    print(encoded_df.head())
    # Data Normalization
    normalizer = StandardScaler()
    dataset_arr = normalizer.fit_transform(encoded_df)
    print(dataset_arr)

    #Slit the data
    X, y = dataset_arr [: , 1:] , dataset_arr [: , 0]

    #Seperate the data into training and validation
    test_size = 0.3 #Ratio
    random_state = 1 
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    #Train the model Random Forest
    regressor = RandomForestRegressor(random_state=random_state)
    regressor.fit(X_train, y_train)

    # #Train the model AdaBoost
    # regressor = AdaBoostRegressor(random_state=random_state)
    # regressor.fit(X_train, y_train)

    # #Train the model Gradient Boosting
    # regressor = GradientBoostingRegressor(random_state=random_state)
    # regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_val)
    mae = mean_absolute_error(y_val,y_pred)
    mse = mean_squared_error (y_val,y_pred)
    print ('Evaluation results on validation set :')
    print (f'Mean Absolute Error : {mae}')
    print (f'Mean Squared Error : {mse}')