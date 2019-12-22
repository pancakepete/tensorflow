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

train_stats = dataset
train_labels = dataset.pop("SalePrice")