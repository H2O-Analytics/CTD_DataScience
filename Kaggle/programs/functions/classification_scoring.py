"""
Name: random_forest_class.py

Purpose: Scores different classification models

Developer: Taylor Waters

Input:  

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
import matplotlib.pyplot as plt
import numpy as np

def gen_scoring(model, y_val, pred, pred_prob):
    '''Creates ROC Curve plot, accuracy measures, and confussion matrix plot'''
    [fpr, tpr, thr] = roc_curve(y_val, pred_prob)
    # check classification scores
    print('Train/Test split results:')
    print(model.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_val, pred))
    print(model.__class__.__name__+" log_loss is %2.3f" % log_loss(y_val, pred_prob))
    print(model.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

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