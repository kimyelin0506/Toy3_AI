from itertools import count

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt  # 시각화 패키지
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd

# 1. 파일 읽기
df = pd.read_csv("../renttherunway_data.csv")

# 2. 유저별 구매량(리뷰 수) 세기
user_purchase_counts = df['user_id'].value_counts().reset_index()
user_purchase_counts.columns = ['user_id', 'purchase_count']

# # 3. CSV로 저장
# user_purchase_counts.to_csv("user_purchase_counts.csv", index=False)
#
# print("user_purchase_counts.csv 파일 저장 완료!")


user = pd.read_csv("./user_purchase_counts.csv")
print(user)
print(user['purchase_count'].sum())
