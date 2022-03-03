import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


df = pd.DataFrame(None, columns = ['path', 'label'])
train_dir = '/opt/ml/input/data/train/images'
out_path = '/opt/ml/code/df.csv'
def age_group(x):
    return min(2, x // 30)

train_df = pd.read_csv("/opt/ml/input/data/train/train.csv")
# print(train_df)

for index, line in tqdm(enumerate(train_df.iloc)):
    for file in list(os.listdir(os.path.join(train_dir, line['path']))):
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
            'path': os.path.join(train_dir, line['path'], file),
            'label': mask * 6 + gender * 3 + age_group(line['age'])
        }
        df = df.append(data, ignore_index=True)

print(df)

# df.to_csv(out_path)