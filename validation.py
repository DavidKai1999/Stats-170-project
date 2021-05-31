import pandas as pd
from model import *
from preprocess import *
from evaluation import *
from joblib import load
import matplotlib.pyplot as plt
from evaluation import *
from WMVE import *
import pickle
from sklearn import preprocessing

def main():

    with open("news_data.txt", "rb") as fp:  # Pickling
        X_train, Y_train, X_test, Y_test, X_val, Y_val = pickle.load(fp)

    #with open("comments_data.txt", "rb") as fp:  # Pickling
    #    X_train_c, Y_train_c, X_test_c, Y_test_c, X_val_c, Y_val_c = pickle.load(fp)

    X = X_train.copy()
    X.extend(X_val)
    scaler = preprocessing.StandardScaler().fit(X)
    X_train_1 = scaler.transform(X_train)
    X_test_1 = scaler.transform(X_test)
    X_val_1 = scaler.transform(X_val)

    #train_comment_voting(X_train_c,Y_train_c)

    train_voting(X_train,X_train_1, X_test, X_test_1, Y_train, Y_test)

    validate(X_val, X_val_1, Y_val)


def train_comment_voting(X_train_c,Y_train_c):
    # ========================================
    #           Train Comment Voting
    # ========================================
    print('Training Comment Voting...')

    # load comment model
    bert_c = torch.load(save_comment_model)
    forest_c = load(forest_comment_model)
    nb_c = load(nb_comment_model)
    lr_c = load(lr_comment_model)

    with open("comments_train_pred.txt", "rb") as fp:  # Pickling
        bert_train_c = pickle.load(fp)
    forest_train_c = forest_c.predict(X_train_c)
    nb_train_c = nb_c.predict(X_train_c)
    lr_train_c = lr_c.predict(X_train_c)

    classfiers_pred_train_c = pd.DataFrame({'bert': bert_train_c,
                                            'forest': forest_train_c,
                                            'nb': nb_train_c,
                                            'lr': lr_train_c})

    # Train voting for comments
    weight_c = train_weight(classfiers_pred_train_c, Y_train_c)
    print("Weight for comment:",weight_c)

    with open("comment_weight.txt", "wb") as fp:  # Pickling
        pickle.dump(weight_c, fp)

    print('Training Comment Voting Complete!')
    print('')


def train_voting(X_train, X_train_1, X_test, X_test_1, Y_train, Y_test):
    # ========================================
    #               Train Voting
    # ========================================
    print('Training Voting...')

    # load model
    #bert = torch.load(save_news_model)
    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    #with open("comment_weight.txt", "rb") as fp:  # Unpickling
    #    weight_c = pickle.load(fp)

    with open("news_train_pred.txt", "rb") as fp:  # Pickling
        bert_train_pred = pickle.load(fp)
    forest_train_pred = forest.predict(X_train)
    nb_train_pred = nb.predict(X_train_1)
    lr_train_pred = lr.predict(X_train_1)
    comment_train_pred = comments_voting(mode='train')

    binary_eval('bert_train', Y_train, bert_train_pred)
    binary_eval('forest_train', Y_train, forest_train_pred)
    binary_eval('nb_train', Y_train, nb_train_pred)
    binary_eval('lr_train', Y_train, lr_train_pred)

    with open("news_test_pred.txt", "rb") as fp:  # Pickling
        bert_test_pred = pickle.load(fp)
    forest_test_pred = forest.predict(X_test)
    nb_test_pred = nb.predict(X_test_1)
    lr_test_pred = lr.predict(X_test_1)
    comment_test_pred = comments_voting(mode='train')

    classfiers_pred_train = pd.DataFrame({'bert': bert_test_pred,
                                          'forest': forest_test_pred,
                                          'nb': nb_test_pred,
                                          'lr': lr_test_pred,
                                          'comment':comment_test_pred})

    weight1, weight2 = train_weight(classfiers_pred_train, Y_test, final=True)
    print('Weight1:',weight1)
    print('Weight2:',weight2)

    with open("voting_weight.txt", "wb") as fp:  # Pickling
        pickle.dump([weight1,weight2], fp)

    voting_train_pred, _ = WMVEpredict([weight1, weight2], classfiers_pred_train,use_softmax=False,final=True)
    binary_eval('voting_train', Y_test, voting_train_pred)

    print('Training Voting Complete!')
    print('')

def validate(X_val,X_val_1,Y_val):

    # ========================================
    #                 Validation
    # ========================================
    # load model
    #bert = torch.load(save_news_model)
    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    with open("voting_weight.txt", "rb") as fp:  # Unpickling
        weight = pickle.load(fp)

    #with open("comment_weight.txt", "rb") as fp:  # Unpickling
    #    weight_c = pickle.load(fp)

    print('Validating...')

    # prediction
    with open("news_val_pred.txt", "rb") as fp:  # Pickling
        bert_val_pred = pickle.load(fp)
    forest_val_pred = forest.predict(X_val)
    nb_val_pred = nb.predict(X_val_1)
    lr_val_pred = lr.predict(X_val_1)
    comment_val_pred = comments_voting(mode='val')

    classfiers_pred_val = pd.DataFrame({'bert': bert_val_pred,
                                        'forest': forest_val_pred,
                                        'nb': nb_val_pred,
                                        'lr': lr_val_pred,
                                        'comment':comment_val_pred})

    voting_val_pred, prob = WMVEpredict(weight, classfiers_pred_val,use_softmax=False,final=True)


    # applying evaluation metrics
    print('')
    binary_eval('bert_val', Y_val, bert_val_pred)
    binary_eval('forest_val', Y_val, forest_val_pred)
    binary_eval('nb_val', Y_val, nb_val_pred)
    binary_eval('lr_val', Y_val, lr_val_pred)
    binary_eval('voting_val', Y_val, voting_val_pred)

    plot_roc(prob,Y_val)
    plt.show()

    print('Complete!')


if __name__ == '__main__':
    main()