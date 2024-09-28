import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

# Điều chỉnh giá trị cột 'target' để gom các giá trị 2, 3, 4 thành 1 (người bị bệnh tim)
df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
# Tăng kích thước biểu đồ để hiển thị rõ hơn
plt.figure(figsize=(12, 8))
sns.set_context('paper', font_scale=1, rc={
                'font.size': 3, 'axes.labelsize': 15, 'axes.titlesize': 10})
ax = sns.catplot(kind='count', data=df, x='age', hue='target',
                 palette='tab10', order=df['age'].sort_values().unique())
ax.ax.set_xticks((np.arange(0, 80, 5)))
# Cài đặt tiêu  đề và nhãn trục
plt.title('Number of patients by Age and Heart Disease Status', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

# Hiển thị biểu đồ
# plt.legend(title='Target (0 = No Heart Disease, 1 = Heart Disease)', fontsize=12)
plt.show()
