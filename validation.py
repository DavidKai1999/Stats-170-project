import pandas as pd
from model import *
from preprocess import *
from config import *
from evaluation import *
from joblib import load
import matplotlib.pyplot as plt
from evaluation import *

from ignite.contrib.metrics import RocCurve

from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix


def main():

    # XY of news features
    X_train, Y_train, X_val, Y_val, train_dataloader, validation_dataloader = get_news_data()

    # XY of comments features
    X_train_c, Y_train_c, X_val_c, Y_val_c, comment_train, comment_val = get_comments_data()

    # ========================================
    #                 Validation
    # ========================================

    # load_model
    bert = torch.load(save_news_model)

    forest = load(forest_model_name)
    nb = load(nb_model_name)
    lr = load(lr_model_name)

    # prediction

    forest_train_pred = forest.predict(X_train)
    forest_val_pred = forest.predict(X_val)

    nb_train_pred = nb.predict(X_train)
    nb_val_pred = nb.predict(X_val)

    lr_train_pred = lr.predict(X_train)
    lr_val_pred = lr.predict(X_val)

    # applying evaluation metrics
    print('')
    binary_eval('forest_train', Y_train, forest_train_pred)
    binary_eval('forest_val', Y_val, forest_val_pred)
    binary_eval('nb_train', Y_train, nb_train_pred)
    binary_eval('nb_val', Y_val, nb_val_pred)
    binary_eval('lr_train', Y_train, lr_train_pred)
    binary_eval('lr_val', Y_val, lr_val_pred)

    plot_roc_curve(forest, X_val, Y_val)
    plot_roc_curve(nb, X_val, Y_val)
    plt.show()


if __name__ == '__main__':
    main()
