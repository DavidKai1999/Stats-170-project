import pandas as pd
from model import *
from preprocess import *
from evaluation import *
from joblib import load
import matplotlib.pyplot as plt
from evaluation import *
from WMVE import *
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
start_time = time.time()

def main():

    with open("news_data.txt", "rb") as fp:  # Pickling
        X_train, Y_train, X_val, Y_val = pickle.load(fp)


    X = X_train.copy()
    X.extend(X_val)
    scaler = preprocessing.StandardScaler().fit(X)
    X_train_1 = scaler.transform(X_train)
    X_val_1 = scaler.transform(X_val)

    temp = gbdt_predict(X_train_1, Y_train, X_val_1, Y_val)


    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    forest_train_pred = forest.predict(X_train)
    nb_train_pred = nb.predict(X_train_1)
    lr_train_pred = lr.predict(X_train_1)

    binary_eval('forest_train', Y_train, forest_train_pred)
    binary_eval('nb_train', Y_train, nb_train_pred)
    binary_eval('lr_train', Y_train, lr_train_pred)

    print('validation')

    forest_val_pred = forest.predict(X_val)
    nb_val_pred = nb.predict(X_val_1)
    lr_val_pred = lr.predict(X_val_1)

    binary_eval('gdbt_val', Y_val, temp)
    binary_eval('forest_val', Y_val, forest_val_pred)
    binary_eval('nb_val', Y_val, nb_val_pred)
    binary_eval('lr_val', Y_val, lr_val_pred)

def gbdt_predict(X_train,y_train,X_test,y_test):

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train)

    n_estimator = 30

    grd = GradientBoostingClassifier(n_estimators=n_estimator)

    grd_enc = OneHotEncoder()

    grd_lm = LogisticRegression()

    grd.fit(X_train, y_train)

    # y_pred_grd = grd.predict_proba(X_test)[:, 1]
    y_pred_grd = grd.predict(X_test)
    fpr_grd, tpr_grd, _ = metrics.roc_curve(y_test, y_pred_grd)
    roc_auc = metrics.auc(fpr_grd, tpr_grd)
    print('predict', roc_auc)

    grd_enc.fit(grd.apply(X_train)[:, :, 0])

    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_lm = grd_lm.predict(grd_enc.transform(grd.apply(X_test)[:, :, 0]))#[:, 1]

    fpr_grd_lm, tpr_grd_lm, _ = metrics.roc_curve(y_test, y_pred_grd_lm)
    roc_auc = metrics.auc(fpr_grd_lm, tpr_grd_lm)

    print("AUC Score :", (metrics.roc_auc_score(y_test, y_pred_grd_lm)))
    print("--- %s seconds ---" % (time.time() - start_time))

    return y_pred_grd_lm

if __name__ == '__main__':
    main()
