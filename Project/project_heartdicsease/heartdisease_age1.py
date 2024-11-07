import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

plt.figure(figsize=(10, 6))

sns.catplot(kind='bar',data=df, x='sex', y='age', hue='target', ci=None, palette='tab10')

plt.title('Distribution of age vs sex with the target class', fontsize=16)
plt.xlabel('Sex (0 = Female, 1 = Male)', fontsize=14)
plt.ylabel('Average Age', fontsize=14)

plt.legend(title='Target (0 = No Heart Disease, 1 = Heart Disease)', fontsize=12)
plt.show()
