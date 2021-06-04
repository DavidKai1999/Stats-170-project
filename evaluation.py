'''This file prints result of evaluation metrics'''
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def binary_eval(model_type:str, label, prediction):
    print(model_type,'accuracy', balanced_accuracy_score(label, prediction))
    print(model_type,'f1',f1_score(label, prediction, average='weighted'))
    print(model_type,'confusion matrix', confusion_matrix(label, prediction))
    print('')

def binary_eval_comment(model_type:str, label, prediction):
    print(model_type,'accuracy', balanced_accuracy_score(label, prediction))
    print(model_type,'f1',f1_score(label, prediction, average='weighted'))
    print(model_type,'confusion matrix', confusion_matrix(label, prediction,labels=[0,1]))
    print('')