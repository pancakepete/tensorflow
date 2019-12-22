from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

dataset_path = os.path.abspath("C:/Users/pwitk/Downloads/train.csv")
test_dataset_path = os.path.abspath("C:/Users/pwitk/Downloads/test.csv")
raw_dataset = pd.read_csv(dataset_path,  sep=",")
raw_test_dataset = pd.read_csv(test_dataset_path, sep=",")

dataset = raw_dataset
dataset.pop("Id")

test_dataset = raw_test_dataset
test_dataset.pop("Id")
test_data = test_dataset

train_data = dataset
train_labels = dataset.pop("SalePrice")

## replacing nulls in train data
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
train_data.select_dtypes(include=['int', 'int32', 'int64', 'float']).isnull().sum()[train_data.select_dtypes(include=['int', 'int32', 'int64', 'float']).isnull().sum() > 0]

for col in ('LotFrontage', 'MasVnrArea', 'GarageYrBlt'):
    train_data[col] = train_data[col].fillna(0);

## replacing nulls in test data
# categorical
test_data.select_dtypes(include='object').isnull().sum()[test_data.select_dtypes(include='object').isnull().sum() > 0]

for col in ('MSZoning', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PoolQC', 'Fence', 'MiscFeature', 'KitchenQual'):
    test_data[col] = test_data[col].fillna('NA')

for col in ('MasVnrType', 'Electrical', 'Exterior1st', 'Exterior2nd', 'Utilities'):
    test_data[col] = test_data[col].fillna('None')

test_data['Functional'] = test_data['Functional'].fillna('Typ')
test_data['SaleType'] = test_data['SaleType'].fillna('Oth')

# numerical
test_data.select_dtypes(include=['int', 'int32', 'int64', 'float']).isnull().sum()[test_data.select_dtypes(include=['int', 'int32', 'int64', 'float']).isnull().sum() > 0]

for col in ('LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars',
            'GarageArea'):
    test_data[col] = test_data[col].fillna(0);

## changing numerical data to categorical:
train_data['MSSubClass'] = train_data['MSSubClass'].astype(str)
test_data['MSSubClass'] = test_data['MSSubClass'].astype(str)