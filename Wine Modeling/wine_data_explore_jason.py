#Importing required packages.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Input data sets
DATA_PATH = "G:/.shortcut-targets-by-id/1zYOUecIzsLDVZFqSsvw6zZiBkCJY67pY/CDT_Data_Science/data_sets/Kaggle/Red Wine Quality/"
wine_df = pd.read_csv(DATA_PATH + "wineQualityReds.csv")

# EXPLORATORY ANALYSIS

#Drop Qaulity#
wine_df = wine_df.drop(columns=['quality'], axis=1)

#Data Set Information
wine_df.head()
wine_df.info()
wine_df.columns

# Check for missing values
wine_df.isna().sum()

# Descriptive Stats for each feature
wine_df.describe()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed.acidity', data = wine_df)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile.acidity', data = wine_df)

fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'citric.acid', data = wine_df)

fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'free.sulfur.dioxide', data = wine_df)

fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'alcohol', data = wine_df)

fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'sulphates', data = wine_df)

# Calculate correlation between each pair of variable
corr_matrix=wine_df.corr()

# plot it
sns.heatmap(corr_matrix, cmap='PuOr')

# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# plot it
sns.heatmap(corr_matrix, mask = mask, cmap = cmap)

#Check correleation between the variables using Seaborn's pairplot. 
sns.pairplot(wine_df)

#count of each target variable
Counter(wine_df['quality'])

#count of the target variable
sns.countplot(x='quality', data=wine_df)

#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed.acidity', data = wine_df)
sns.boxplot('quality', 'volatile.acidity', data = wine_df)
sns.boxplot('quality', 'citric.acid', data = wine_df)
sns.boxplot('quality', 'residual.sugar', data = wine_df)
sns.boxplot('quality', 'chlorides', data = wine_df)
sns.boxplot('quality', 'free.sulfur.dioxide', data = wine_df)
sns.boxplot('quality', 'total.sulfur.dioxide', data = wine_df)
sns.boxplot('quality', 'density', data = wine_df)
sns.boxplot('quality', 'pH', data = wine_df)
sns.boxplot('quality', 'sulphates', data = wine_df)
sns.boxplot('quality', 'alcohol', data = wine_df)