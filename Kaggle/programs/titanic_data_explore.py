import pandas as pd
import numpy as np
import os

# Data input
DATA_PATH_TW = "/Users/tawate/Library/CloudStorage/OneDrive-SAS/CDT_DataScience_Drive/kaggle_ds/titanic/"
train_df = pd.read_csv(DATA_PATH_TW + "train.csv")
test_df = pd.read_csv(DATA_PATH_TW + "test.csv")

# Exploratory Analysis
# Check for missing values
train_df.isna().sum()
# Random Forest Prediction
