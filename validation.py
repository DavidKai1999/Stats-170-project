import pandas as pd
from model import *
from preprocess import *
from config import *
from evaluation import *
from joblib import load
import matplotlib.pyplot as plt
from evaluation import *
from WMVE import *

from ignite.contrib.metrics import RocCurve

from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix

def main():

    # XY of news features
    X_train, Y_train, X_val, Y_val, \
    train_inputs, train_masks, validation_inputs, validation_masks, _, _ = get_news_data()

    # XY of comments features
    #X_train_c, Y_train_c, X_val_c, Y_val_c,\
    #train_inputs_c, train_masks_c, validation_inputs_c, validation_masks_c = get_comments_data()


    # load_model
    bert = torch.load(save_news_model)

    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    # ========================================
    #               Train Voting
    # ========================================

    bert_train_pred = bertpredict(bert, train_inputs, train_masks)
    forest_train_pred = forest.predict(X_train)
    nb_train_pred = nb.predict(X_train)
    lr_train_pred = lr.predict(X_train)

    binary_eval('forest_train', Y_train, forest_train_pred)
    binary_eval('nb_train', Y_train, nb_train_pred)
    binary_eval('lr_train', Y_train, lr_train_pred)

    classfiers_pred = pd.DataFrame({'bert': bert_train_pred,
                                    'forest': forest_train_pred,
                                    'nb': nb_train_pred,
                                    'lr': lr_train_pred})

    weight = train_weight(classfiers_pred, Y_train)
    voting_train_pred = WMVEpredict(weight, classfiers_pred,use_softmax=True)
    binary_eval('voting_train', Y_train, voting_train_pred)


    # ========================================
    #                 Validation
    # ========================================
    '''
    # prediction
    bert_val_pred = bertpredict(bert, validation_inputs,validation_masks)
    forest_val_pred = forest.predict(X_val)
    nb_val_pred = nb.predict(X_val)
    lr_val_pred = lr.predict(X_val)

    classfiers_pred = pd.DataFrame({'bert': bert_val_pred,
                                    'forest': forest_val_pred,
                                    'nb': nb_val_pred,
                                    'lr': lr_val_pred})


    
    # applying evaluation metrics
    print('')   
    binary_eval('forest_val', Y_val, forest_val_pred)
    binary_eval('nb_val', Y_val, nb_val_pred)
    binary_eval('lr_val', Y_val, lr_val_pred)
    plot_roc_curve(forest, X_val, Y_val)
    plot_roc_curve(nb, X_val, Y_val)
    plt.show()
    '''


if __name__ == '__main__':
    main()
