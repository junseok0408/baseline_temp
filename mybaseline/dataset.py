# 마스크 데이터셋을 읽고 전처리를 진행한 후 데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일입니다.
# 이 곳에서, 나만의 Data Augmentation 기법 들을 구현하여 사용할 수 있습니다.

# torch.utils.data의 Dataset 라이브러리 상속

import math
import sys

import os
import pandas as pd
from PIL import Image



from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


import numpy as np
from scipy import stats
import seaborn as sns
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("/opt/ml/input/data/train/train.csv")
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

preprocessing_df = pd.DataFrame(columns=['gender','age', 'class'])



# class MyDataset(Dataset) :
#     # MyDataset 클래스가 처음 선언 되었을 때 호출
#     def __init__(self):
#         pass
#
#     # MyDataset의 데이터 중 index 위치의 아이템을 리턴
#     def __getitem__(self, index):
#         return None
#
#     # MyDataset 아이템의 전체 길이
#     def __len__(self):
#         return None

# # (Batch, Channel, Height, Width)
# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size = batch_size,
#     num_workers=num_workers,
#     drop_last=True,
# )
