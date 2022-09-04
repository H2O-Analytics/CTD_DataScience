import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Data input
DATA_PATH_TW = "/Users/tawate/Library/CloudStorage/OneDrive-SAS/CDT_DataScience_Drive/kaggle_ds/titanic/"
train_df = pd.read_csv(DATA_PATH_TW + "train.csv")
test_df = pd.read_csv(DATA_PATH_TW + "test.csv")

# Exploratory Analysis
# Check for missing values
train_df.isna().sum()

# Descriptive Stats for each feature
train_df.describe()

# Feature analysis and engineering
train_df.loc[train_df['Age'].isnull(),'Age'] = 0
train_df = train_df.drop(columns=['Cabin'])
train_df.loc[train_df['Embarked'].isnull(),'Embarked'] = 'NA'
train_df.isna().sum()

# One Hot enonding
train_enc = pd.get_dummies(train_df)

# convert to arrays and remove labels from features
labels = np.array(train_enc['Survived'])
features = train_enc.drop('Survived', axis=1)
feature_list = list(features.columns)
features = np.array(features)

# Split training ds into traing and validation set
feature_train, feature_test, label_train, label_test = train_test_split(features, 
labels, test_size=.25, random_state=42)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(feature_train, label_train)

# predictions on test set
preds = rf.predict(feature_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(label_test, preds))
