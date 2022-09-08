"""
Name: titanic_logistic_reg.py

Purpose: Create end to end logistic regression to predict the suvival of a passenger on the titanic

Developer: Taylor Waters

Input:  train.csv:  titanic survival training set
        test.csv:   titanc survival test set without predictor var

Output

Parameters:
        DATA_PATH_TW:   path for TW google drive with kaggle titanic datasets

Usage

History:
Date        User    Ticket #    Description
08SEP2022   TW      ITKTP-11    | Initial Developement

"""
# Import Packages
from ast import increment_lineno
from difflib import IS_LINE_JUNK
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Input data sets
DATA_PATH_TW = "/Users/tawate/My Drive/CDT_Data_Science/data_sets/Kaggle/Titanic/"
train_df = pd.read_csv(DATA_PATH_TW + "train.csv")
test_df = pd.read_csv(DATA_PATH_TW + "test.csv")

"""
Exploratory Analysis
    1. Overall descriptive statistics
    2. Check for missing values
    3. Check feature distributions with target
    4. Check target distribution
"""
# Info on each feature
train_df.info()

# Check target distribution
sns.countplot(x = 'Survived', data=train_df, palette='hls')
plt.show()
pct_surv = len(train_df[train_df['Survived']==1]) / len(train_df)
pct_no_surv = len(train_df[train_df['Survived']==0]) / len(train_df)
print("pct survived = ", pct_surv*100)
print("pct not survived = ", pct_no_surv*100)

# Percent missing values by feature
train_df.isnull().sum() * 100 / len(train_df)

# Check means of numeric vars by target and categorical vars
    # 1. Higher class and higher fare passengers had better chance of survival
train_df.groupby('Survived').mean()
train_df.groupby('Sex').mean()
train_df.groupby('Cabin').mean()
train_df.groupby('Ticket').mean()
train_df.groupby('Embarked').mean()

# Distribution of each categorical variable with target
    # 1. Males were more likely to die vs females
    # 2. Embarking from Cherbourg gave increases liklihood of survival. 
    #    (higher fare prices)
pd.crosstab(train_df.Sex, train_df.Survived).plot(kind = 'bar')
pd.crosstab(train_df.Cabin, train_df.Survived).plot(kind = 'bar')
pd.crosstab(train_df.Ticket, train_df.Survived).plot(kind = 'bar')
pd.crosstab(train_df.Embarked, train_df.Survived).plot(kind = 'bar')

# Historgram distribution of each numeric variables
    # Left skewness to fare price
train_df.Pclass.hist()
train_df.Age.hist()
train_df.SibSp.hist()
train_df.Parch.hist()
train_df.Fare.hist(bins = 50)

"""
Variable Manipulation
    1. Create dummy variables
"""


