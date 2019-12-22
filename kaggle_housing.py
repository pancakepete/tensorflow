from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

dataset_path = os.path.abspath("C:/Users/pwitk/Downloads/train.csv")
raw_dataset = pd.read_csv(dataset_path,  sep=",")

dataset = raw_dataset
dataset.pop("Id")

train_data = dataset
train_labels = dataset.pop("SalePrice")

## replacing nulls
# categorical:
train_data.select_dtypes(include='object').isnull().sum()[train_data.select_dtypes(include='object').isnull().sum() > 0]

for col in ('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PoolQC', 'Fence', 'MiscFeature'):
    train_data[col] = train_data[col].fillna('NA')

for col in ('MasVnrType', 'Electrical'):
    train_data[col] = train_data[col].fillna('None')

# numerical:
train_data.select_dtypes(include=['int', 'float']).isnull().sum()[train_data.select_dtypes(include=['int', 'float']).isnull().sum() > 0]

for col in ('LotFrontage', 'MasVnrArea', 'GarageYrBlt'):
    train_data[col] = train_data[col].fillna(0);

## changing numerical data to categorical:
train_data['MsSubClass']=train_data['MsSubClass'].astype(str)