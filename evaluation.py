'''This file prints result of evaluation metrics'''
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def binary_eval(model_type:str, label, prediction):
    print(model_type,'accuracy', balanced_accuracy_score(label, prediction))
    print(model_type,'f1',f1_score(label, prediction, average='weighted'))
    print(model_type,'roc_auc', roc_auc_score(label, prediction, average='weighted'))
    print(model_type,'confusion matrix', confusion_matrix(label, prediction))
    print('')

def binary_eval_comment(model_type:str, label, prediction):
    print(model_type,'accuracy', balanced_accuracy_score(label, prediction))
    print(model_type,'f1',f1_score(label, prediction, average='weighted'))
    print(model_type,'confusion matrix', confusion_matrix(label, prediction,labels=[0,1]))
    print('')


def plot_roc(pred, Y):
    ns = [0 for _ in range(len(pred))]

    fpr, tpr, _ = roc_curve(Y, ns, pos_label=1)
    voting_fpr, voting_tpr, _ = roc_curve(Y, pred, pos_label=1)

    plt.plot(fpr, tpr, linestyle='--', label='No Skill Line')
    plt.plot(voting_fpr, voting_tpr, marker='.', label='Weighted Voting')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
