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
        4. https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook
History:
Date        User    Ticket #    Description
08SEP2022   TW      ITKTP-11    | Initial Developement
09SEP2022   TW      ITKTP-11    | Included variable manipulation and dummy variable creation
26SEP2022   TW      ITKTP-11    | Test log reg assumptions. Do recursive feature selection. Fit final model.
30SEP2022   TW      ITKTP-2     | Wrap redundant code into modeling and model scoring functions. See programs/fucntion/* for
                                  details.
05OCT2022   TW/JT   ITKTP-30    | Included profession type using name titles
"""
# Import Packages
from re import L
from tabnanny import verbose

from pytest import param
from functions.classification_models import *
from functions.classification_scoring import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.genmod import families
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
from hyperopt import hp
import xgboost as xgb

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
    1. Variable imputation
    2. Drop trouble variables
    3. Feaure engineering
    4. Create dummy variables
"""
# List of titles by type
professional_titles = ['Dr','Rev']
military_titles = ['Col','Major','Capt']
royalty_titles = ['Master','Sir','Lady','Mme','Don','Jonkheer','the Countess','Countess']

# Title to Profession_Type mapping
train_df['name_title'] = train_df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_df['profession_type'] = train_df.name_title.apply(lambda x : 'professional' if x in professional_titles else
                                                        ('military' if x in military_titles else
                                                        ('royalty' if x in royalty_titles else
                                                        'working')))
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median(skipna=True))
train_df = train_df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId','name_title'])
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
print(train_df.isnull().sum()*100/len(train_df))

# Create dummy variables
cat_cols = ['Sex','Embarked','Pclass','profession_type']
# Remove first variable to prevent coliniearity
train_df_onehot = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
# Create feature and label data frames
X = train_df_onehot.drop(columns=['Survived'])
X_const = sm.add_constant(train_df_onehot.drop(columns=['Survived']))
y = train_df_onehot['Survived']

"""
Testing Assumptions
    1. Check for independence of variables and log odds for continous variables(Box - Tidwell)
    2. Check for influential outliers
    3. Check for multicolinearity
    4. Check for independence of observations
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
    # Log_Age p value < .05 therefore Ha therefore linearity not violated
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
# We are looking for absence of trend in the below plot. Enough to eye it.
logit_model = GLM(y, X_const, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
# Generate residual series plot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                    xlabel="Index Number", ylabel="Deviance Residuals")

ax.plot(X.index.tolist(), stats.zscore(logit_results.resid_deviance))
plt.axhline(y=0, ls="--", color='red')

"""
Feature Selection
    1. Create Fare^2 var
    2. Advanced technique: recursive feature elimination
"""
# Create Fare^2 and drop Fare based on linearity assumption
X['Fare^2'] = X['Fare'] * X['Fare']
X = X.drop(columns=['Fare'])
# Recursive feature elimination
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
Selected_Features = list(X.columns[rfecv.support_])


"""
Model Fitting and Scoring
    1. Split into train and validation
    2. Create regression model with selected features
    3. Assess model fit and model diagnositics
    4. Calculate odds ratios
    5. Create ROC curve
    6. Perform cross validation
"""
# Split into train and val
X = X[Selected_Features]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.20, random_state=None)

# Hyper Parameter Tunning Using HYPEROPT Package
# import hyperparameter tuning packages
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, anneal

# define the hyper parameter space
# 1. quiform: rounds decimal values and returns an integer from low to high
# 2. uniform: uniform values between low and high
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform("n_estimators",50,200,10),
        'learning_rate': hp.loguniform("learning_rate",0,1),
        'seed': 0
    }
 
 # define the objective function
def objective(space):
    xgb_mod=xgb.XGBClassifier(  n_estimators =int(space['n_estimators']), 
                                max_depth = int(space['max_depth']), 
                                gamma = int(space['gamma']),
                                reg_alpha = int(space['reg_alpha']),
                                min_child_weight=int(space['min_child_weight']),
                                colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_val, y_val)]
    
    xgb_mod.fit(X_train, y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10,verbose=False)
    

    pred = xgb_mod.predict(X_val)
    accuracy = accuracy_score(y_val, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

# apply optmization algorithm
trials = Trials()
best_hyp_params_tpe = fmin( fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)
best_hyp_params_anneal = fmin( fn=objective,
                        space=space,
                        algo=anneal.suggest,
                        max_evals=100,
                        trials=trials)

# print best hyper parameters
print(best_hyp_params_tpe)
print(best_hyp_params_anneal)


# XBG model with Random Search hyper parameter tuning
xgb_model = xgb.XGBClassifier(  objective='binary:logistic',
                                booster='gbtree',
                                eval_metric='auc',
                                tree_method='hist',
                                grow_policy='lossguide',
                                use_label_encoder=False)
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train)
xgb_model.get_params()

# Hyperparameter Tuning using RandomSearch
# Define the search space
param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,6,7,8,9,10,11,12,13,14],
              'n_estimators': [50,65,80,100,115,130,150],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
# Define the scoring metric to optimize
scoring = ['accuracy']

# K-fold cross validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)

random_search = RandomizedSearchCV( estimator=xgb_model,
                                    param_distributions=param_grid,
                                    n_iter = 48,
                                    scoring=scoring,
                                    refit='accuracy',
                                    n_jobs=-1,
                                    cv=kfold,
                                    verbose=0)

random_result = random_search.fit(X_train, y_train)
# Print the best score and the corresponding hyperparameters
print(f'The best score is {random_result.best_score_:.4f}')
print('The best score standard deviation is', round(random_result.cv_results_['std_test_accuracy'][random_result.best_index_], 4))
print(f'The best hyperparameters are {random_result.best_params_}')

# Apply hyper parameters to XGBoost
xgb_mod = xgb.XGBClassifier(objective='binary:logistic',
                            booster='gbtree',
                            eval_metric='auc',
                            tree_method='hist',
                            grow_policy='lossguide',
                            use_label_encoder=False,
                            reg_lambda = 3.2,
                            reg_alpha = .4,
                            n_estimators = 80,
                            max_depth = 14,
                            learning_rate = .25,
                            gamma = .8)
xgb_mod.fit(X_train, y_train)
xgb_pred = xgb_mod.predict(X_val)
xgb_pred_prob = xgb_mod.predict_proba(X_val)[:,1]

# XGB Scoring
gen_scoring(model=xgb_mod,
            y_val=y_val,
            pred=xgb_pred,
            pred_prob=xgb_pred_prob)




""" Notes on XGBoosting and Hyper Parameter Tuning

XGBoosting Hyper Parameters:
3 Sets of Hyper Parameters
    1. General Parameters: guide overall function of xgboost
        i. booster: defines the booster to use
            - gbtree: tree based 
            - gblinear: linear function based
            - dart: tree based
        ii. verbosity: used to print different types of messages (will not affect performance of model)
            - 0: no message
            - 1: warnings
            - 2: info
            - 3: debug
        iii. nthread: number of parrell threads to use (computer dependent)
    2. Booster Parameters: guide which booster we should be using
        i. eta or learning rate
            - defines a weight that correponds to the step size shrinkage of new featres after each boosting step.
            - makes model more robust
            - helps prevent over fitting
        ii. gamma or min_split_loss
            - specifies the min loss reduction required to make a node split
            - varies depending on the loss function
        iii. max_depth
            - max depth of the tree
            - helps prevent overfitting since large depth size correlate to overfitting
            - high values correlate to longer run run times
            - tuned using cross validation
        iv. min_child_weight
            - defines the minimum sum of weights of all observations required in a child (whats a child)
        v. max_depth_step
            - 
        vi. subsample
        vii. colsample_bytree, bylevel, bynode
        viii. lambda
        viv. alpha
        x. tree_method
        xi. scale_pos_weight
        xii. max_leaves
    3. Learning Task Parameters: define the optmization objective or metric to be calculated at each step
        i. objective
        ii. eval_metric
        iii. seed


HyperOPT: Python hyper parameter optimization package
    - describe objective function to minimize
    - space to search (grid)
    - database/list of all possible point evaluations
    - search algorithm (randomsearch etc.)



"""