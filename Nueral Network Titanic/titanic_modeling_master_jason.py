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

# Define the search space (testing using hyperopt, needs more workhttps://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook)
"""
param_grid={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform("n_estimators",10,200,10),
        'seed': 0}
"""
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

# Random forest model
rf_mod, rf_pred, rf_pred_prob, rf_importance  = rand_forest(n_estimators=1000,
                                                    x_train=X_train,
                                                    y_train=y_train,
                                                    x_val=X_val,
                                                    random_state=None)
# RF Scoring
gen_scoring(model=rf_mod,
            y_val=y_val,
            pred=rf_pred,
            pred_prob=rf_pred_prob)

# Log reg cross validation scoring
cv_scoring( mod=rf_mod,
            X=X,
            y=y,
            cv=10)

# Log Reg Model
log_mod, log_pred, log_pred_prob = log_reg( max_iter = 1000,
                                            log_x_train=X_train,
                                            log_y_train=y_train,
                                            log_x_val=X_val)

# Log Reg Scoring
gen_scoring(model=log_mod,
            y_val=y_val,
            pred=log_pred,
            pred_prob=log_pred_prob)


# Odds ratio for log reg
odds_ratio( model=log_mod,
            x_train=X_train)

# Log reg cross validation scoring
cv_scoring( mod=log_mod,
            X=X,
            y=y,
            cv=10)


" SVM Model"
svm_model = svm.classifier()
svm_predict = svm_model.pred
svm_predict_prob = svm_model.pred_prob

gen_scoring(model = svm_model,
y_val=y_val,
pred= svm_pred,
pred_prob = svm_pred_prob)

"""
Precision: TP/(TP+FP) determines the accuracy of positive predictions
Recall: TP/(TP+FN) determines the fraction of positives that were correctly identified
F1 Score: is a mean of precision and recall. Best F1 = 1, worst = 0    
"""

#Import Packages Jason Nueral Networks
import numpy as np 
import pandas as pd
import tensorflow as tf

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.neural_network import MLPClassifier



# Reproductibility
from numpy.random import seed
seed(1002)
tf.random.set_seed(1002)

# Input data sets
DATA_PATH = "G:/.shortcut-targets-by-id/1zYOUecIzsLDVZFqSsvw6zZiBkCJY67pY/CDT_Data_Science/data_sets/Kaggle/Titanic/"
train_df = pd.read_csv(DATA_PATH + "train.csv")
test_df = pd.read_csv(DATA_PATH + "test.csv")

# Adding a column in each dataset before merging
train_df['Type'] = 'train'
test_df['Type'] = 'test'

# Merging train and test
data = train_df.append(test_df)

####################################
# Missing values and new features
####################################

# Title
data['Title'] = data['Name']

# Cleaning name and extracting Title
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
# Replacing rare titles 
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 
           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}
           
data.replace({'Title': mapping}, inplace=True)
titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']

# Replacing missing age by median/title 
for title in titles:
    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute

    
# New feature : Family_size
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1
data.loc[:,'FsizeD'] = 'Alone'
data.loc[(data['Family_Size'] > 1),'FsizeD'] = 'Small'
data.loc[(data['Family_Size'] > 4),'FsizeD'] = 'Big'

# Replacing missing Fare by median/Pclass 
fa = data[data["Pclass"] == 3]
data['Fare'].fillna(fa['Fare'].median(), inplace = True)

#  New feature : Child
data.loc[:,'Child'] = 1
data.loc[(data['Age'] >= 18),'Child'] =0

# New feature : Family Survival (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
DEFAULT_SURVIVAL_VALUE = 0.5

data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
                               
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                    
####################################
# Encoding and pre-modeling
####################################                  

# dropping useless features
data = data.drop(columns = ['Age','Cabin','Embarked','Name','Last_Name',
                            'Parch', 'SibSp','Ticket', 'Family_Size'])

# Encoding features
target_col = ["Survived"]
id_dataset = ["Type"]
cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()
cat_cols   = [x for x in cat_cols ]

# numerical columns
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + id_dataset]

# Binary columns with 2 values
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])

# Duplicating columns for multi value columns
data = pd.get_dummies(data = data,columns = multi_cols )

# Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns = num_cols)

# dropping original values merging scaled values for numerical columns
df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index = True,right_index = True,how = "left")
data = data.drop(columns = ['PassengerId'],axis = 1)

# Target = 1st column
cols = data.columns.tolist()
cols.insert(0, cols.pop(cols.index('Survived')))
data = data.reindex(columns= cols)

# Cutting train and test
train = data[data['Type'] == 1].drop(columns = ['Type'])
test = data[data['Type'] == 0].drop(columns = ['Type'])

# X and Y
X_train = train.iloc[:, 1:20].to_numpy()
y_train = train.iloc[:,0].to_numpy()

####################################
# Keras - Neural Networks
####################################

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim = 18, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn = create_baseline, epochs = 20, batch_size = 10, verbose = 1)
kfold = StratifiedKFold(n_splits = 5, random_state = (None), shuffle = False)
results = cross_val_score(estimator, X_train, y_train, cv = kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# X Test
X_test = test.iloc[:, 1:20].to_numpy()

estimator.fit(X_train, y_train, epochs = 20, batch_size = 10)

# Predicting y_test
prediction = estimator.predict(X_test).tolist()

# List to series
data_check =  df_test
se = pd.Series(prediction)

# Creating new column of predictions in data_check dataframe
data_check['check'] = se
data_check['check'] = data_check['check'].str.get(0)

series = []
for val in data_check.check:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
data_check['final'] = series

match = 0
nomatch = 0
for val in data_check.values:
    if val[1] == val[3]:
        match = match +1
    else:
        nomatch = nomatch +1

####################################
# Submission
#################################### 

temp = pd.DataFrame(test_df['PassengerId'])
temp['Survived'] = data_check['final']
temp.to_csv("G:/.shortcut-targets-by-id/1zYOUecIzsLDVZFqSsvw6zZiBkCJY67pY/CDT_Data_Science/data_sets/nn_jason/working.csv", index = False)

"Nueral Network Model"
nn_model = neural_network.classifier()
nn_pedict = nn_model.pred
nn_pedict_prob = nn_model.pred_prob

gen_scoring(model = nn_model,
y_val=y_val,
pred= nn_pred,
pred_prob = nn_pred_prob)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))



####################################
# Second Nureal Network Model (Jason)
####################################

# data processing
import numpy as np
import pandas as pd 

# machine learning
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import *

# utils
import time
from datetime import timedelta

# some configuratin flags and variables
verbose=0 # Use in classifier

# defeine random seed for reproducibility
seed = 69
np.random.seed(seed)

# Input data sets
DATA_PATH = "G:/.shortcut-targets-by-id/1zYOUecIzsLDVZFqSsvw6zZiBkCJY67pY/CDT_Data_Science/data_sets/Kaggle/Titanic/"
train_df = pd.read_csv(DATA_PATH + "train.csv", index_col='PassengerId')
test_df = pd.read_csv(DATA_PATH + "test.csv", index_col='PassengerId')

# Show the columns
train_df.columns.values

# Show the shape
train_df.shape

# preview the training dara
train_df.head()

# Show that there is NaN data (Age,Fare Embarked), that needs to be handled during data cleansing
train_df.isnull().sum()

def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    
    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
      
    # Convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(enbarked_one_hot)

    return df

train_df = prep_data(train_df)
train_df.isnull().sum()

# X contains all columns except 'Survived'  
X = train_df.drop(['Survived'], axis=1).values.astype(float)

# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df['Survived'].values

def create_model(optimizer='adam', init='uniform'):
    # create model
    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

run_gridsearch = False

if run_gridsearch:
    
    start_time = time.time()
    if verbose: print (time.strftime( "%H:%M:%S " + "GridSearch started ... " ) )
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 200, 400]
    batches = [5, 10, 20]
    
    model = KerasClassifier(build_fn=create_model, verbose=verbose)
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if verbose: 
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        elapsed_time = time.time() - start_time  
        print ("Time elapsed: ",timedelta(seconds=elapsed_time))
        
    best_epochs = grid_result.best_params_['epochs']
    best_batch_size = grid_result.best_params_['batch_size']
    best_init = grid_result.best_params_['init']
    best_optimizer = grid_result.best_params_['optimizer']
    
else:
    # pre-selected paramters
    best_epochs = 200
    best_batch_size = 5
    best_init = 'glorot_uniform'
    best_optimizer = 'rmsprop'

# Create a classifier with best parameters
model_pred = KerasClassifier(build_fn=create_model, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose)
model_pred.fit(X, Y)

# Prep and clean data
test_df = prep_data(test_df)

# Create X_test
X_test = test_df.values.astype(float)

# Scaling
X_test = scale.transform(X_test)

# Predict 'Survived'
prediction = model_pred.predict(X_test)

#Save Predictions
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction[:,0],
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-simple-cleansing.csv', index=False)

# Read the data
df_train = pd.read_csv(DATA_PATH + "train.csv",index_col='PassengerId')
df_test = pd.read_csv(DATA_PATH + "train.csv",index_col='PassengerId')  
l = len(df_train.index)
    

## All data train and test in one dataframe 
dfa = df_train.append(df_test)

# Drop unwanted features
dfa = dfa.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
# Fill missing data: Fare with mean, Embarked with most frequent value
dfa[['Fare']] = dfa[['Fare']].fillna(value=dfa[['Fare']].mean())
dfa[['Embarked']] = dfa[['Embarked']].fillna(value=dfa['Embarked'].value_counts().idxmax())
    
# Convert categorical features into numeric
dfa['Sex'] = dfa['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Convert 'Embarked' to one-hot
embarked_one_hot = pd.get_dummies(dfa['Embarked'], prefix='Embarked')
dfa = dfa.drop('Embarked', axis=1)
dfa = dfa.join(embarked_one_hot)

dfa.head()

# Split data in to training set (Age not null) and 'to-be-predicted' set (Age in nan)
df_age_train = dfa[dfa.Age.notnull()]
df_age_nan = dfa[dfa.Age.isnull()]

# split data into input X and output Y
X = df_age_train.drop(['Age', 'Survived'], axis=1).values.astype(float)
Y = df_age_train['Age'].values.astype(float)

X_test = df_age_nan.drop(['Age', 'Survived'], axis=1).values.astype(float)

def age_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Create a Pipeline

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=age_model, epochs=100, batch_size=5, verbose=verbose)))
pipeline = Pipeline(estimators)

# Cross-validation
kfold = KFold(n_splits=2, random_state=seed, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Result: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# Predict
pipeline.fit(X, Y)
prediction_train = pipeline.predict(X)
prediction_test = pipeline.predict(X_test)

# Create a data frame with PassengerId and predicted age
df_age_pred = pd.DataFrame({
    'PassengerId': df_age_nan.index,
    'Age_pred': prediction_test.astype(int)
})
df_age_pred.set_index('PassengerId', inplace=True)

# Add column with predicted age to the dataframe with all data (dfa)
dfa2 = df_train.append(df_test) 
dfa_pred = pd.concat([dfa2, df_age_pred], axis=1)   

# Update Age column with prediction where nan and remove Age_pred
dfa_pred['Age'] = np.where(pd.isnull(dfa_pred['Age']), dfa_pred['Age_pred'] , dfa_pred['Age'])
dfa_pred = dfa_pred.drop(['Age_pred'], axis=1)

# Create new files
l = len(df_train)
df_train2 = dfa_pred[0:l] 
df_test2 = dfa_pred[l:] 
df_test2 = df_test2.drop(['Survived'], axis=1)

df_train2.to_csv('train-age-predicted.csv')
df_test2.to_csv('test-age-predicted.csv')