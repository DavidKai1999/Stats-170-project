'''This file prints result of evaluation metrics'''
from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score

def binary_eval(model_type:str, label, prediction):
    print(model_type,'accuracy', balanced_accuracy_score(label, prediction))
    print(model_type,'forest f1',f1_score(label, prediction, average='weighted'))
    print(model_type,'roc_auc', roc_auc_score(label, prediction, average='weighted'))
    print(model_type,'confusion matrix', confusion_matrix(label, prediction))
    print('')
