"""
Name: classification_models.py
Purpose: Creates classification model results
Developer: Taylor Waters
Input:  X_train - training feature data
        y_train - training target data
        max iter
        n_estimator
        x_val
Output
Parameters:
Usage:
Resources Used:
History:
Date        User    Ticket #    Description
30SEP2022   tawate  ITKTP-22    Initial funciton development
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def rand_forest(n_estimators, random_state, x_train, y_train, x_val):
    '''Takes in input train and validation data and generates a random forest classifier object
       and prediction array'''
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_val)
    rf_pred_prob = rf_model.predict_proba(x_val)[:,1]
    rf_importance = rf_model.feature_importances_
    return rf_model, rf_pred, rf_pred_prob, rf_importance


def log_reg(max_iter, log_x_train, log_y_train, log_x_val):
    '''Create log reg model and predicition output'''
    log_model = LogisticRegression(max_iter=max_iter)
    log_model.fit(log_x_train, log_y_train)
    log_pred = log_model.predict(log_x_val)
    log_pred_prob = log_model.predict_proba(log_x_val)[:, 1]
    return log_model, log_pred, log_pred_prob
    