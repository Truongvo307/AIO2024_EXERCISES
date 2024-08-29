import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

dataset_path = 'IMDB-Movie-Data.csv'
data = pd.read_csv(dataset_path)
print(data)

data_indexed = pd.read_csv( dataset_path , index_col ="Title")
# Preview top 5 rows using head ()
print(data.head())
print(data.info())
print(data.describe())

genre = data['Genre']
print(genre)

# Extract data as dataframe
data[['Genre']]
some_cols = data [['Title','Genre','Actors','Director','Rating']] #mutilple columns

data.iloc [5:15][['Title' ,'Rating' ,'Revenue (Millions)']]
print(data)


#select data by condition 
se_data = data[((data['Year'] >= 2010) & (data['Year'] <= 2015)) & ( data['Rating'] < 6.0) & ( data['Revenue (Millions)'] > data['Revenue (Millions)'].quantile(0.95))]
print(f'selection data {se_data}')

#group data 
gr_data = data.groupby('Director') [['Rating']].mean().head()
print(f'Group by data {gr_data}')

#sort data
sort_data = data.groupby('Director')[['Rating']].mean().sort_values(['Rating'],ascending=False ).head()
print(f'Sorted data {sort_data}')

# To check null values row - wise
print(data.isnull().sum())

# Use drop function to drop columns
print(f'Before drop {data.columns}')
data_dr = data.drop('Metascore', axis =1).head()
print(f'After drop {data_dr.columns}')

#Dealing with missing values - Filling:
revenue_mean = data_indexed['Revenue (Millions)'].mean()
print ("The mean revenue is: ", revenue_mean)
# We can fill the null values with this mean revenue
data_indexed['Revenue (Millions)'].fillna(revenue_mean, inplace=True)
print(data_indexed.isnull().sum())

# Classify movies based on ratings
def rating_group ( rating ) :
    if rating >= 7.5:
        return 'Good' 
    elif rating >= 6.0:
        return 'Average' 
    else: return 'Bad' 
    
# Lets apply this function on our movies data
# creating a new variable in the dataset to hold the rating category
data['Rating_category'] = data['Rating'].apply(rating_group)
print(data[['Title','Director','Rating','Rating_category']].head(5))


