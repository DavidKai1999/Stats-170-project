import pandas as pd
from model import *
from preprocess import *
from config import *


def main():
    # XY of news features
    #X_train, Y_train, X_val, Y_val, \
    #train_inputs, train_masks, validation_inputs, validation_masks,\
    #train_dataloader, validation_dataloader = get_news_data()

    # XY of comments features
    X_train_c, Y_train_c, X_val_c, Y_val_c, _, _, _, _, comment_train, comment_val = get_comments_data()

    # ========================================
    #                Classifiers
    # ========================================

    # train bert model, model save in 'news/comments + bertmodel.h5'
    #bertpretrain(train_dataloader, validation_dataloader,'news')
    bertpretrain(comment_train, comment_val,'comment')


    # train other model (random forest / SVM / Naive Bayes/ ... )

    #foresttrain(X_train, Y_train, forest_news_model)
    #nbtrain(X_train, Y_train, nb_news_model)
    #lrtrain(X_train, Y_train, lr_news_model)

    #foresttrain(X_train_c, Y_train_c, forest_comment_model)
    #nbtrain(X_train_c, Y_train_c, nb_comment_model)
    #lrtrain(X_train_c, Y_train_c, lr_comment_model)

    print('Classifiers Training Complete!')


if __name__ == '__main__':
    main()