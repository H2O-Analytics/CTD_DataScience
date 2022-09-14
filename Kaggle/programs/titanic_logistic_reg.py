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

Resources Used:
        1. https://www.kaggle.com/competitions/titanic/data?select=train.csv
        2. https://github.com/kennethleungty/Logistic-Regression-Assumptions/blob/main/Logistic_Regression_Assumptions.ipynb
        3. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
        
History:
Date        User    Ticket #    Description
08SEP2022   TW      ITKTP-11    | Initial Developement
09SEP2022   TW      ITKTP-11    | Included variable manipulation and dummy variable creation
"""
# Import Packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Input data sets
DATA_PATH = "/Users/tawate/My Drive/CDT_Data_Science/data_sets/Kaggle/Titanic/"
train_df = pd.read_csv(DATA_PATH + "train.csv")
test_df = pd.read_csv(DATA_PATH + "test.csv")

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
    1. Impute median of age for missing age
    2. Remove Cabin, too many missing values. Remove name (does't matter and too sparse)
    3. Impute mode of Embarked for missing Embarked
    4. Create dummy variables
"""
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median(skipna=True))
train_df = train_df.drop(columns=['Cabin', 'Name', 'Ticket'])
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
print(train_df.isnull().sum()*100/len(train_df))

# Create dummy variables
#   Ticket is sparse, condider more investigation here
cat_cols = ['Sex','Embarked','Pclass']
train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True) # Remove first variable to prevent coliniearity
train_features = train_df.drop(columns=['Survived'])
train_label = train_df['Survived']

"""
Testing Assumptions
"""
# only works for positive values
train_df2 = train_df.drop(train_df[train_df['Age'] == 0].index)
train_df2 = train_df2.drop(train_df[train_df['Fare'] == 0].index)

"""
Feature Selection

"""
import statsmodels.api as sm
model = sm.Logit(endog=train_label, exog=train_features).fit()
print(model.summary())