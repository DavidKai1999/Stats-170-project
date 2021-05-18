import pandas as pd
from model import *
from preprocess import *
from evaluation import *
from joblib import load
import matplotlib.pyplot as plt
from evaluation import *
from WMVE import *

from sklearn.metrics import plot_roc_curve, roc_curve

def main():
    # XY of news features
    X_train, Y_train, X_val, Y_val, \
    train_inputs, train_masks, validation_inputs, validation_masks,_,_ = get_news_data()

    # XY of comments features
    X_train_c, Y_train_c, X_val_c, Y_val_c,\
    train_inputs_c, train_masks_c, validation_inputs_c, validation_masks_c,_,_ = get_comments_data()

    # load model
    bert = torch.load(save_news_model)
    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)


    # ========================================
    #           Train Comment Voting
    # ========================================
    print('Training Comment Voting...')
    print('')
    # load comment model

    bert_c = torch.load(save_comment_model)
    forest_c = load(forest_comment_model)
    nb_c = load(nb_comment_model)
    lr_c = load(lr_comment_model)

    bert_train_c = bertpredict(bert_c, train_inputs_c, train_masks_c)
    forest_train_c= forest_c.predict(X_train_c)
    nb_train_c = nb_c.predict(X_train_c)
    lr_train_c = lr_c.predict(X_train_c)

    classfiers_pred_train_c = pd.DataFrame({'bert': bert_train_c,
                                    'forest': forest_train_c,
                                    'nb': nb_train_c,
                                    'lr': lr_train_c})

    # Train voting for comments
    weight_c = train_weight(classfiers_pred_train_c, Y_train_c)

    # ========================================
    #               Train Voting
    # ========================================
    print('Training Voting...')
    print('')

    bert_train_pred = bertpredict(bert, train_inputs, train_masks)
    forest_train_pred = forest.predict(X_train)
    nb_train_pred = nb.predict(X_train)
    lr_train_pred = lr.predict(X_train)
    comment_train_pred = comments_voting(weight_c,mode='train')

    binary_eval('bert_train', Y_train, bert_train_pred)
    binary_eval('forest_train', Y_train, forest_train_pred)
    binary_eval('nb_train', Y_train, nb_train_pred)
    binary_eval('lr_train', Y_train, lr_train_pred)

    classfiers_pred_train = pd.DataFrame({'bert': bert_train_pred,
                                          'forest': forest_train_pred,
                                          'nb': nb_train_pred,
                                          'lr': lr_train_pred,
                                          'comment':comment_train_pred})

    #weight = train_weight(classfiers_pred_train, Y_train)
    weight1, weight2 = train_weight(classfiers_pred_train, Y_train, final=True)
    print(weight1,weight2)
    voting_train_pred = WMVEpredict([weight1, weight2], classfiers_pred_train,use_softmax=False,final=True)
    binary_eval('voting_train', Y_train, voting_train_pred)

    '''
    # ========================================
    #                 Validation
    # ========================================
    print('Validating...')
    print('')

    # prediction
    bert_val_pred = bertpredict(bert, validation_inputs,validation_masks)
    forest_val_pred = forest.predict(X_val)
    nb_val_pred = nb.predict(X_val)
    lr_val_pred = lr.predict(X_val)
    #comment_val_pred = comments_voting(weight_c,mode='val')

    classfiers_pred_val = pd.DataFrame({'bert': bert_val_pred,
                                    'forest': forest_val_pred,
                                    'nb': nb_val_pred,
                                    'lr': lr_val_pred})

    #voting_val_pred = WMVEpredict([weight], classfiers_pred_val,use_softmax=False)
    voting_val_pred = WMVEpredict([weight1,weight2], classfiers_pred_val,use_softmax=False,final=True)

    
    # applying evaluation metrics
    print('')
    binary_eval('bert_val', Y_val, bert_val_pred)
    binary_eval('forest_val', Y_val, forest_val_pred)
    binary_eval('nb_val', Y_val, nb_val_pred)
    binary_eval('lr_val', Y_val, lr_val_pred)
    binary_eval('voting_val', Y_val, voting_val_pred)

    #plot_roc_curve(forest, X_val, Y_val)
    #plot_roc_curve(nb, X_val, Y_val)
    roc_curve(Y_val, voting_val_pred,pos_label=1)
    plt.show()'''



if __name__ == '__main__':
    main()