import pandas as pd
from model import *
from config import *
from joblib import load
import matplotlib.pyplot as plt

from sklearn.metrics import plot_roc_curve, accuracy_score

# ========================================
#             Import Dataset
# ========================================
user = 'postgres'
password = 'Komaeda'

from sqlalchemy import create_engine
engine = create_engine('postgresql://'+user+':'+password+'@localhost/news')

Query = "SELECT * FROM redditcomment"
comment_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT * FROM redditnews"
news_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT * FROM factcheck"
factcheck = pd.read_sql_query(Query, con=engine)


# ========================================
#                 Main
# ========================================

def main():
    redditnews = news_table
    sample = redditnews.sample(n=100,random_state=1)

    redditcomment = comment_table

    reddit = pd.merge(sample,redditcomment,
                      how='left',
                      left_on=['title','text'],
                      right_on=['title','text'])
    comment_notnull = reddit.dropna(subset=['comment_text'])

    text = sample.text.values
    label = sample.label.values
    comment = comment_notnull.comment_text.values
    comment_label = comment_notnull.label.values


    attention_masks, input_ids = vectorize(text)  # tokenization + vectorization
    attention_masks_comment, input_ids_comment = vectorize(comment)

    X_train, Y_train, X_val, Y_val, train_dataloader, validation_dataloader = vector_to_input(attention_masks,
                                                                                              input_ids,
                                                                                              label)

    X_train_c, Y_train_c, X_val_c, Y_val_c, comment_train, comment_text = vector_to_input(attention_masks_comment,
                                                                                          input_ids_comment,
                                                                                          comment_label)


    # ========================================
    #                 Train
    # ========================================

    # train bert model, model save in 'news/comments + bertmodel.h5'
    #bertpretrain(train_dataloader, validation_dataloader,'news')
    #bertpretrain(comment_train, comment_text,'comment')

    # train other model (random forest / SVM / Naive Bayes/ ... )

    # generate model file
    foresttrain(X_train, Y_train, forest_model_name)
    nbtrain(X_train, Y_train, nb_model_name)



    # ========================================
    #                 Validation
    # ========================================

    # load_model
    forest_prediction = forest_predict(X_val, forest_model_name)
    nb_prediction = nb_predict(X_val, nb_model_name)

    # applying evaluation metrics
    print('forest accuracy', accuracy_score(Y_val, forest_prediction))
    print('nb accuracy', accuracy_score(Y_val, nb_prediction))

    forest = load(forest_model_name)
    nb = load(nb_model_name)
    plot_roc_curve(forest, X_val, Y_val)
    plot_roc_curve(nb, X_val, Y_val)
    plt.show()



if __name__ == '__main__':
    main()
