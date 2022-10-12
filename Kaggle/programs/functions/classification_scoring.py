"""
Name: random_forest_class.py
Purpose: Scores different classification models
Developer: Taylor Waters
Fucntions:
        gen_scoring: produces general classification scoring metrics
        odds_ratio: outputs odds ratios for each feature in a logistic regression
Output
Parameters:
Usage:
Resources Used:
History:
Date        User    Ticket #    Description
30SEP2022   tawate  ITKTP-22    Initial funciton development
"""
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def gen_scoring(model, y_val, pred, pred_prob):
    '''Creates ROC Curve plot, accuracy measures, and confussion matrix plot'''
    [fpr, tpr, thr] = roc_curve(y_val, pred_prob)
    # Print classification scores
    print('Train/Test split results:')
    print(model.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_val, pred))
    print(model.__class__.__name__+" log_loss is %2.3f" % log_loss(y_val, pred_prob))
    print(model.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
    #classification report
    print(classification_report(y_val, pred))

    # Plot Confussion Matrix
    cf_matrix = confusion_matrix(y_val, pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()

    # index of the first threshold for which the sensibility > 0.95
    idx = np.min(np.where(tpr > 0.95))
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], color='blue')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], color='blue')
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


def cv_scoring(mod, X, y, cv):
    '''
    Inputs:
        mod = model objet
        X = features
        y = labels
        cv = number of cross validations
    Outputs: 
        {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
    Notes:
        1. Uses cross_val_score function
        2. We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
    '''
    scores_accuracy = cross_val_score(mod, X, y, cv=cv, scoring='accuracy')
    scores_log_loss = cross_val_score(mod, X, y, cv=cv, scoring='neg_log_loss')
    scores_auc = cross_val_score(mod, X, y, cv=cv, scoring='roc_auc')
    print('K-fold cross-validation results:')
    print('Number for cross-validations: ' + str(cv))
    print(mod.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
    print(mod.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
    print(mod.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())


def odds_ratio(model, x_train):
    '''
    calculate the odds ratios for logistic regression model only
    model = logistic regression model object
    x_train = training features data set
    '''
    np.exp(model.coef_)
    for index, var in enumerate(x_train.columns):
        print(var + " : " + str(np.exp(model.coef_)[0][index]))