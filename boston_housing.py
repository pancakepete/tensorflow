import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_boston

# load Data
# Load Dataset
boston = load_boston()

# Print out the Dataset
print(boston)

features_df = pd.DataFrame(np.array(boston.data), columns=[boston.feature_names])
