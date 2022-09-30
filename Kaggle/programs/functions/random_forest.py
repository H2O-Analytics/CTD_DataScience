"""
Name: random_forest_class.py

Purpose: Creates a generic random forest classification model depending on user input

Developer: Taylor Waters

Input:  X_train - training feature data
        y_train - training target data


Output

Parameters:

Usage:

Resources Used:

History:
Date        User    Ticket #    Description
30SEP2022   tawate  ITKTP-22    Initial funciton development
"""
from sklearn.ensemble import RandomForestClassifier
def rand_forest(n_estimators, random_state, X_train, y_train, X_val):
    '''Takes in input train and validation data and generates a random forest classifier object
       and prediction array'''
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    return rf_pred, rf_model
