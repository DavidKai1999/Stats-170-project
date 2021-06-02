import pandas as pd
from model import *
from preprocess import *
from config import *
import pickle
from sklearn import preprocessing


def main():

    with open(".\\tempfile\\news_data.txt", "rb") as fp:  # Pickling
        X_train, Y_train, X_test, Y_test, X_val, Y_val = pickle.load(fp)

    labels = Y_train.tolist().copy()
    labels.extend(Y_test.tolist())
    labels.extend(Y_val.tolist())

    neg = labels.count(0)
    pos = labels.count(1)
    total = neg + pos

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # ========================================
    #                Classifiers
    # ========================================

    # train other model (random forest / SVM / Naive Bayes/ ... )

    X = X_train.copy()
    X.extend(X_val)
    scaler = preprocessing.StandardScaler().fit(X)
    X_train_1 = scaler.transform(X_train)

    foresttrain(X_train, Y_train, forest_news_model,class_weight)
    nbtrain(X_train_1, Y_train, nb_news_model)
    lrtrain(X_train_1, Y_train, lr_news_model,class_weight)

    print('Classifiers Training Complete!')


if __name__ == '__main__':
    main()