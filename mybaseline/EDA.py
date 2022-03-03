import os
import math
import sys

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from PIL import Image
import torch.utils.data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("/opt/ml/input/data/train/train.csv")
# print(train.head(10))

### ??. 사람들의 나이가 너무 어리게 되어있다...??

########################################################
### 0. 기본정보
## 데이터의 총 길이는 2700
## columns = ['id', 'gender', 'race', 'age', 'path']
# print(train.describe(include='all'))

########################################################
### 1. ['id'] 요상한 id가 있다?? & 중복된 id가 있다??

# for i, value in enumerate(train.values) :
#     try :
#         temp = int(value[0])
#     except :
#         print(value)

# print(train['id'].value_counts())

# print(train[train['id'] == '003397'])
########################################################
### 2. ['gender'] 여성이 더 많다!!
## female = 1658, male = 1042
# print(train['gender'].value_counts())
## 아래는 그래프
# sns.histplot(x='gender',data=train)
# plt.show()

########################################################
### 3. ['race'] : 아시아인 뿐이다... 이건 df에서 제거해도 될듯...
# print(train['race'].value_counts())

########################################################
### 4. ['age'] : 30대, 40대의 데이터가 매우 부족하다....
## 연령별 히스토그램을 보자
sns.histplot(x='age',data=train, )
plt.show()

## 그렇다면 클래스대로 자르면 어떨까?
## 연령 분류 = ['30세미만','30세이상60세미만', '60세이상']
train['age_class'] = str()
for i, value in enumerate(train['age']) :
    if value < 30 :
        train['age_class'][i] = '30세미만'
    elif (30 <= value < 60) :
        train['age_class'][i] = '30세이상60세미만'
    elif 60 <= value :
        train['age_class'][i] = '60세이상'
    else :
        print('오류')

# print(train.head(10))

# sns.histplot(x='age_class',data=train, )
# plt.show()


age_gender = train.groupby('gender')['age_class'].value_counts().sort_index()
print(age_gender)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

idx = np.arange(len(age_gender['male'].index))
width = 0.35

ax.bar(idx - width / 2, age_gender['male'],
       color='royalblue',
       width=width, label='Male')

ax.bar(idx + width / 2, age_gender['female'],
       color='tomato',
       width=width, label='Female')

ax.set_xticks(idx)
ax.set_xticklabels(age_gender['male'].index)
ax.legend()

plt.show()

########################################################
### 5. ['path']
## 앞선 column을 모두 합친거라서 별로 특별한 건 없다.