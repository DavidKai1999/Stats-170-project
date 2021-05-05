import pandas as pd
from model import *
from config import *
from joblib import load
import matplotlib.pyplot as plt

from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix

# ========================================
#             Import Dataset
# ========================================
user = 'postgres'
password = 'Komaeda'
topic_file = "C:\\Users\\gyiko\\OneDrive - personalmicrosoftsoftware.uci.edu\\STATS\\STATS 170AB\\Project\\datasets\\Sample_Topic.csv"

from sqlalchemy import create_engine
engine = create_engine('postgresql://'+user+':'+password+'@localhost/news')

Query = "SELECT * FROM redditcomment"
comment_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT title,text,label FROM redditnews"
news_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT title,text,author,label FROM factcheck"
factcheck = pd.read_sql_query(Query, con=engine)

topic = pd.read_csv(topic_file)


# ========================================
#                 Main
# ========================================
def main():

    reddit_sample = news_table.sample(n=100,random_state=1)
    factcheck_sample = factcheck.sample(n=10,random_state=1)

    # Combine factcheck dataset and redditcomment dataset
    news = pd.merge(reddit_sample,factcheck_sample,
                    how='outer',
                    left_on=['title','text','label'],
                    right_on=['title','text','label']).reset_index(drop=True)

    news_with_topic = pd.merge(news,topic,
                               how='left',
                               left_on=['title', 'text'],
                               right_on=['title', 'text']).reset_index(drop=True)

    print('Length of news:', len(news_with_topic))

    df = pd.merge(news_with_topic,comment_table,
                      how='left',
                      left_on=['title','text'],
                      right_on=['title','text']).reset_index(drop=True)

    print('Length after merging with comment:', len(df))

    comment_notnull = df.dropna(subset=['comment_text']).reset_index(drop=True)

    text = news.text.values
    label = news.label.values
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
    forest = load(forest_model_name)
    nb = load(nb_model_name)

    # prediction
    forest_train_pred = forest.predict(X_train)
    forest_val_pred = forest.predict(X_val)

    nb_train_pred = nb.predict(X_train)
    nb_val_pred = forest.predict(X_val)

    # applying evaluation metrics
    print('')
    print('forest train accuracy', accuracy_score(Y_train, forest_train_pred))
    print('forest train matrix \n',confusion_matrix(Y_train,forest_train_pred))
    print('forest test accuracy', accuracy_score(Y_val, forest_val_pred))
    print('forest test matrix \n',confusion_matrix(Y_val,forest_val_pred))

    print('')
    print('nb train accuracy', accuracy_score(Y_train, nb_train_pred))
    print('nb train matrix \n', confusion_matrix(Y_train, nb_train_pred))
    print('nb test accuracy', accuracy_score(Y_val, nb_val_pred))
    print('nb test matrix \n', confusion_matrix(Y_val, nb_val_pred))


    plot_roc_curve(forest, X_val, Y_val)
    plot_roc_curve(nb, X_val, Y_val)
    plt.show()


if __name__ == '__main__':
    main()