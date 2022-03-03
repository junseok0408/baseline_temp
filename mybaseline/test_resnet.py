import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
import pandas as pd

'''
신규범님 코드 참고

학습 데이터 구축
'''

def age_group(x):
    if x < 30:
        return 0
    elif x < 60:
        return 1
    else:
        return 2


df = []
train_df = pd.read_csv("/opt/ml/input/data/train/train.csv")
image_data_dir = '/opt/ml/input/data/train/images'

for idx, line in tqdm(enumerate(train_df.iloc)):
    for file in list(os.listdir(os.path.join(image_data_dir, line['path']))):
        if file[0] == '.':
            continue
        if file.split('.')[0] == 'normal':
            mask = 2
        elif file.split('.')[0] == 'incorrect_mask':
            mask = 1
        else:
            mask = 0
        gender = 0 if line['gender'] == 'male' else 1
        data = {
            'id': line['id'],
            'gender': line['gender'],
            'age_group': age_group(line['age']),
            'mask': mask,
            'path': os.path.join(image_data_dir, line['path'], file),
            'label': mask * 6 + gender * 3 + age_group(line['age'])
        }
        df.append(data)

df = pd.DataFrame(df)

'''
ResNet 모델 구축

나동빈님 코드 참고

https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/ResNet18_MNIST_Train.ipynb

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


# ResNet18을 위해 최대한 간단히 수정한 BasicBlock 클래스 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 3x3 필터를 사용 (너비와 높이를 줄일 때는 stride 값 조절)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)

        # 3x3 필터를 사용 (패딩을 1만큼 주기 때문에 너비와 높이가 동일)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential()  # identity인 경우
        if stride != 1:  # stride가 1이 아니라면, Identity mapping이 아닌 경우
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # (핵심) skip connection
        out = F.relu(out)
        return out


# ResNet 클래스 정의
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes  # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


'''
Sample_submission 코드 참고

데이터 셋 구축
'''


class CustomDataset(Dataset):
    def __init__(self, df, transform, train=True):

        image_dir = '../input/data/eval/images'

        self.train = train
        self.df = df
        if self.train:
            self.img_paths = self.df['path'].tolist()
            self.labels = self.df['label'].tolist()
        else:
            self.img_paths = [os.path.join(image_dir, img_id) for img_id in self.df.ImageID]
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, torch.tensor(self.labels[index])
        else:
            return image

    def __len__(self):
        return len(self.img_paths)


'''
학습 함수 설정
'''


def train(model, data_loder, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, targets) in tqdm(enumerate(data_loder)):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = model(images)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss /= len(data_loder)
    acc = correct / total

    return train_loss, acc


def val(model, data_loder, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(data_loder):
        with torch.no_grad():
            images, targets = images.to(device), targets.to(device)
            benign_outputs = model(images)
            loss = criterion(benign_outputs, targets)
            val_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(data_loder)
    acc = correct / total

    return val_loss, acc


def pred(model, data_loder):
    all_predictions = []
    for images in data_loder:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

    return all_predictions