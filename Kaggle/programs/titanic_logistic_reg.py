"""
Name: titanic_logistic_reg.py

Purpose: Create end to end logistic regression to predict the suvival of a passenger on the titanic

Developer: Taylor Waters

Input:  train.csv:  titanic survival training set
        test.csv:   titanc survival test set without predictor var

Output

Parameters:
        DATA_PATH_TW:   path for TW google drive with kaggle titanic datasets

Usage:

Resources Used:
        1. https://www.kaggle.com/competitions/titanic/data?select=train.csv
        2. https://github.com/kennethleungty/Logistic-Regression-Assumptions/blob/main/Logistic_Regression_Assumptions.ipynb
        3. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
        
History:
Date        User    Ticket #    Description
08SEP2022   TW      ITKTP-11    | Initial Developement
09SEP2022   TW      ITKTP-11    | Included variable manipulation and dummy variable creation
26SEP2022   TW      ITKTP-11    | Test log reg assumptions. Do recursive feature selection. Fit final model.
"""
# Import Packages
from asyncio import threads
from cmath import inf
from distutils.log import Log
from tkinter.font import families
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFECV


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
train_df = train_df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
print(train_df.isnull().sum()*100/len(train_df))

# Create dummy variables
cat_cols = ['Sex','Embarked','Pclass']
# Remove first variable to prevent coliniearity
train_df_onehot = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
# Create feature and label data frames
X = train_df_onehot.drop(columns=['Survived'])
X_const = sm.add_constant(train_df_onehot.drop(columns=['Survived']))
y = train_df_onehot['Survived']
"""
Testing Assumptions
    1. Check for independence of variables and log odds for continous variables(Box - Tidwell)
    2. No influential outliers
"""
# 1. Independece and Box tidwell test
sns.heatmap(train_df_onehot.corr())
# remove non 0 values for continous variables to perform box-tidwell
train_df_onehot = train_df_onehot.drop(train_df[train_df['Age'] == 0].index)
train_df_onehot = train_df_onehot.drop(train_df[train_df['Fare'] == 0].index)
# Define continous variables
cont_var = ['Age', 'Fare']
# Add logit tranformation for each continous variable (var*log(var))
for var in cont_var:
    train_df_onehot[f'{var}:Log_{var}'] = train_df_onehot[var].apply(lambda x: x*np.log(x))
# Keep continous variables
keep_cols = cont_var + train_df_onehot.columns.tolist()[-len(cont_var):]
# Split into feature and labels
train_features = train_df_onehot.drop(columns=['Survived'])
# Include interaction terms
X_lt = train_features[keep_cols]
y_lt = train_df_onehot['Survived']
# Add constant term
X_lt_const = sm.add_constant(X_lt, prepend = False)
# Fit and print generized linear model to data
logit_results = GLM(y_lt, X_lt_const, family=families.Binomial()).fit()
print(logit_results.summary())
# Stat Test:
    # Ho: non linear relationship between var*log(var)
    # Ha: linear relationship between var*log(var)
    # Log_Fare p value > .05 therefore Ho therefore linearity violated
    # Lof_Age p value < .05 therefore Ha therefore linearity not violated
    # Solution is to apply Fare^2 interaction term
# Visualization of Linearity
logit_results2 = GLM(y, X_const, family=families.Binomial()).fit()
pred = logit_results2.predict(X_const)
# log odds
log_odds = np.log(pred / (1 - pred))
# Visual for age
plt.scatter(x = X_const['Age'].values, y = log_odds);
plt.show()
# Visual for fare
plt.scatter(x=X_const['Fare'].values, y = log_odds);
plt.show()

# 2. Check for influential outliers
# logit model
logit_model = GLM(y, X_const, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
# Get influence measures
influence = logit_results.get_influence()
summ_df = influence.summary_frame()
# Filter summary to Cook distance (measure out outlier intesity)
diagnosis_df = summ_df.loc[:,['cooks_d']]
# Append standarized residuals
diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df.loc[:,'std_resid'].apply(lambda x: np.abs(x))
# Sort by Cook's Distance
diagnosis_df.sort_values("cooks_d", ascending=False)
# Cook D threshold
cook_threshold = 4 / len(X)
# Plot Cook'D
fig = influence.plot_index(y_var= "cooks",threshold = cook_threshold)
plt.axhline(y = cook_threshold, ls="--", color = "red")
fig.tight_layout(pad=2)
# Find number of observations that exceed Cook's distance threshold
outliers = diagnosis_df[diagnosis_df['cooks_d'] > cook_threshold]
prop_outliers = round(100*(len(outliers) / len(X)),1)
print(f'Proportion of data points that are highly influential = {prop_outliers}%')
# Find number of observations which are BOTH outlier (std dev > 3) and highly influential
extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & 
                       (diagnosis_df['std_resid'] > 3)]
prop_extreme = round(100*(len(extreme) / len(X)),1)
print(f'Proportion of highly influential outliers = {prop_extreme}%')
# Display top 5 most influential outliers
extreme.sort_values("cooks_d", ascending=False).head()
X.iloc[297]
X.iloc[261]
X.iloc[301]
X.iloc[498]
X.iloc[570]

# 3. Check for multicollinearity
# check correlation matrix
corrMatrix = X.corr()
sns.heatmap(corrMatrix, annot=True, cmap="RdYlGn")
# Variance inflation factor (Threshold is 5)
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)
calc_vif(X_const)
# No significant multi colinearity.
# Note if we did not drop first level for each dummy variable you would see multi colinearity


# 4. Indepednce of observations
# fit model
logit_model = GLM(y, X_const, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
# Generate residual series plot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                    xlabel="Index Number", ylabel="Deviance Residuals")

ax.plot(X.index.tolist(), stats.zscore(logit_results.resid_deviance))
plt.axhline(y=0, ls="--", color='red');
# We are looking for absence of trend in the above plot. Enough to eye it.

"""
Feature Selection
    1. Advanced technique: recursive feature elimination
"""
rfecv = RFECV(estimator=LogisticRegression(max_iter=1000), step=1, cv=10, scoring='accuracy')
rfecv.fit(X,y)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))
# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
Selected_Features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',
                    'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']
"""
Model Fitting
    1. Split into train and validation
    2. Create regression model with selected features
    3. Assess model fit and model diagnositics
    4. Calculate odds ratios
    5. Create ROC curve
    6. Perform cross validation
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
# Split into train and val
X = X[Selected_Features]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.20, random_state=42)

# check classification scores
logreg = LogisticRegression(max_iter = 1000)
model = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
y_pred_prob = logreg.predict_proba(X_val)[:, 1]
[fpr, tpr, thr] = roc_curve(y_val, y_pred_prob)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_val, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_val, y_pred_prob))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

# assess model fit and model stats
logit_model = sm.Logit(y_train, sm.add_constant(X_train))
result = logit_model.fit()
stats1 = result.summary()
stats2 = result.summary2()
print(stats1)
print(stats2)

# calculate odds ratios
model.get_params()
model.intercept_
model.classes_
np.exp(model.coef_)
for index, var in enumerate(X_train.columns):
    print(var + " : " + str(np.exp(model.coef_)[0][index]))

# index of the first threshold for which the sensibility > 0.95
idx = np.min(np.where(tpr > 0.95))

# Plot ROC curve 
plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))

# 10-fold cross-validation logistic regression
logreg = LogisticRegression(max_iter=1000)
# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())