import numpy as np
import pandas as pd
from model import *
from config import *
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from itertools import chain, repeat, islice

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# ========================================
#             Import Dataset
# ========================================

def import_data():
    user = 'postgres'
    password = 'Komaeda'

    from sqlalchemy import create_engine
    engine = create_engine('postgresql://'+user+':'+password+'@localhost/news')

    Query = "SELECT * FROM redditcomment"
    comment_table = pd.read_sql_query(Query, con=engine)
    Query = "SELECT title,text,label FROM redditnews"
    news_table = pd.read_sql_query(Query, con=engine)
    Query = "SELECT title,text,author,label FROM factcheck"
    factcheck = pd.read_sql_query(Query, con=engine)
    Query = "SELECT * FROM topic"
    topic = pd.read_sql_query(Query, con=engine)

    reddit_sample = news_table#.sample(n=500, random_state=1)
    factcheck_sample = factcheck#.sample(n=50, random_state=1)

    # ========================================
    #             Combine Table
    # ========================================
    news = pd.merge(reddit_sample, factcheck_sample,
                    how='outer',
                    left_on=['title', 'text', 'label'],
                    right_on=['title', 'text', 'label']).reset_index(drop=True)

    news_df = pd.merge(news, topic,
                               how='left',
                               left_on=['title', 'text'],
                               right_on=['title', 'text']).reset_index(drop=True)

    news_df = news_df.assign(news_index = [i for i in range(0,len(news_df))])

    comments_df = pd.merge(news_df, comment_table,
                           how='left',
                           left_on=['title', 'text'],
                           right_on=['title', 'text']).dropna(subset=['comment_text']).reset_index(drop=True)
    comments_df = comments_df.assign(have_comment=[1] * len(comments_df))

    news_comments_relationship = comments_df[['news_index', 'have_comment']].drop_duplicates()

    news_df.to_csv('temp_news_df.csv')
    comments_df.to_csv('temp_comments_df.csv')
    news_comments_relationship.to_csv('temp_relationship.csv')
    return news_df, comments_df,news_comments_relationship


def get_news_data():
    news_df, _, _ = import_data()
    print('Length of news:', len(news_df))

    news_df['text_combined'] = news_df['text'] + news_df['title'] * 2
    text_title_combined = news_df.text_combined.values

    attention_masks, input_ids = vectorize(text_title_combined,tokenizer)  # tokenization + vectorization

    news_df = news_df.assign(attention_mask=attention_masks,
                             input_id=input_ids.tolist())  # Assign mask and input_ids back to dataframe



    X_train, Y_train, X_val, Y_val, \
    X_test, Y_test, test_dataloader,\
    train_dataloader, validation_dataloader = vector_to_input(news_df,attention_masks,input_ids,'news',tokenizer)

    return X_train, Y_train, X_val, Y_val, \
           X_test, Y_test, test_dataloader,\
           train_dataloader, validation_dataloader

def get_comments_data():
    #_, comments_df, _ = import_data()
    comments_df = pd.read_csv('temp_comments_df.csv')

    print('Length of comments:', len(comments_df))

    attention_masks, input_ids = vectorize(comments_df.comment_text.values,tokenizer, MAX_LEN=20)
    comments_df.assign(attention_mask=attention_masks,
                       input_id=input_ids.tolist())


    X_train, Y_train, X_val, Y_val, \
    X_test, Y_test, test_dataloader,\
    train_dataloader, validation_dataloader = vector_to_input(comments_df,attention_masks,input_ids,'comments',tokenizer)

    return X_train, Y_train, X_val, Y_val, \
           X_test, Y_test, test_dataloader,\
           train_dataloader, validation_dataloader

# ========================================
#             Helper Functions
# ========================================
def vectorize(text,tokenizer,MAX_LEN=MAX_LEN):
    input_ids = []
    for t in text:
        # so basically encode tokenizing , mapping sentences to thier token ids after adding special tokens.
        try:
            encoded_sent = tokenizer.encode(
                t,  # Sentence which are encoding.
                add_special_tokens=True,  # Adding special tokens '[CLS]' and '[SEP]'
            )
        except:
            encoded_sent = encoded_sent = tokenizer.encode(
                'nan',  # Sentence which are encoding.
                add_special_tokens=True,  # Adding special tokens '[CLS]' and '[SEP]'
            )
        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN , truncating="post", padding="post")

    attention_masks = []
    for sent in input_ids:
        # Generating attention mask for sentences.
        #   - when there is 0 present as token id we are going to set mask as 0.
        #   - we are going to set mask 1 for all non-zero positive input id.
        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)

    return attention_masks, input_ids

def single_word_vec(word,tokenizer):
    encoded_sent = tokenizer.encode(
        word,  # Sentence which are encoding.
        add_special_tokens=True,  # Adding special tokens '[CLS]' and '[SEP]'
    )

    encoded_sent = list(pad(encoded_sent,5,0))

    return encoded_sent

def vector_to_input(df,attention_masks,input_ids,mode,tokenizer):
    labels = df.label.values
    print('Old Labels Count:', np.bincount(labels))

    if mode=='news':
        # X index: input_id, topic, author
        temp = input_ids.tolist()
        author = []
        for a in df['author'].fillna(0):
            if a == 0:
                author.append([0,0,0,0,0])
            else:
                vec = single_word_vec(a,tokenizer)
                author.append(vec)
        for i in range(0,len(df)):
            temp[i].extend(author[i])
            temp[i].append(int(df['topic'][i]))
        X = np.array(temp)

        print('Dealing with imbalanced...')

        over = SMOTE(sampling_strategy=0.4, random_state=1)
        under = RandomUnderSampler(sampling_strategy=0.6, random_state=1)
        resample = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=resample)

        input_ids, _ = pipeline.fit_resample(input_ids, labels)
        attention_masks, _ = pipeline.fit_resample(attention_masks, labels)
        X, labels = pipeline.fit_resample(X, labels)

        train_inputs, validation_inputs, train_labels, validation_labels = k_fold_split(input_ids, labels)
        train_inputs, test_inputs, _, _ = train_test_split(train_inputs, train_labels,
                                                           random_state=1, test_size=0.2)

        train_masks, validation_masks, _, _ = k_fold_split(attention_masks, labels)
        train_masks, test_masks, _, _ = train_test_split(train_masks, train_labels,
                                                         random_state=1, test_size=0.2)

        train_X, val_X, _, _ = k_fold_split(X, labels)
        train_X, test_X, train_labels, test_labels = train_test_split(train_X, train_labels, random_state=1,
                                                                      test_size=0.2)

        # changing the numpy arrays into tensors for working on GPU.
        train_inputs = torch.tensor(train_inputs)
        test_inputs = torch.tensor(test_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        test_masks = torch.tensor(test_masks)
        validation_masks = torch.tensor(validation_masks)

        print('New Labels Count:', np.bincount(labels))

    elif mode=='comments':
        # X index: input_id, comment_author, comment_score, comment_subreddit
        temp = input_ids.tolist()
        mask = attention_masks
        author = []
        author_mask = []
        for a in df['comment_author'].fillna(0):
            if a == 0:
                author.append([0,0,0,0,0])
                author_mask.append([0,0,0,0,0])
            else:
                vec = single_word_vec(a,tokenizer)
                author.append(vec)
                author_mask.append([1, 1, 1, 1, 1])
        subreddit = []
        subreddit_mask = []
        for s in df['comment_subreddit'].fillna(0):
            if s == 0:
                subreddit.append([0, 0, 0, 0, 0])
                subreddit_mask.append([0, 0, 0, 0, 0])
            else:
                vec = single_word_vec(s,tokenizer)
                subreddit.append(vec)
                subreddit_mask.append([1, 1, 1, 1, 1])
        for i in range(0, len(df)):
            temp[i].extend(author[i])
            mask[i].extend(author_mask[i])
            temp[i].extend(subreddit[i])
            mask[i].extend(subreddit_mask[i])
        X = np.array(temp)
        X_mask = np.array(mask)

        train_X_mask, val_X_mask, train_labels, validation_labels = k_fold_split(X_mask, labels)
        train_X_mask, test_X_mask, _, _ = train_test_split(train_X_mask, train_labels,
                                                                                random_state=1, test_size=0.2)

        train_X, val_X, _, _ = k_fold_split(X, labels)
        train_X, test_X, train_labels, test_labels = train_test_split(train_X, train_labels, random_state=1,
                                                                      test_size=0.2)

        # changing the numpy arrays into tensors for working on GPU.
        train_inputs = torch.tensor(train_X)
        test_inputs = torch.tensor(test_X)
        validation_inputs = torch.tensor(val_X)

        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_X_mask)
        test_masks = torch.tensor(test_X_mask)
        validation_masks = torch.tensor(val_X_mask)


    # DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # DataLoader for our testing set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # DataLoader for our validation(test) set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_X, train_labels, val_X, validation_labels, \
           test_X, test_labels, test_dataloader,\
           train_dataloader, validation_dataloader


def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def k_fold_split(X, Y, index=k_index, n_split=n_split):
    kf = KFold(n_splits=n_split,shuffle=True,random_state=1)

    X_trains = []
    X_tests = []
    for train,test in kf.split(X):
        X_train = []
        X_test = []
        for i in train:
            X_train.append(X[i])
        for i in test:
            X_test.append(X[i])
        X_trains.append(X_train)
        X_tests.append(X_test)

    Y_trains = []
    Y_tests = []
    for train, test in kf.split(Y):
        Y_train = []
        Y_test = []
        for i in train:
            Y_train.append(Y[i])
        for i in test:
            Y_test.append(Y[i])
        Y_trains.append(Y_train)
        Y_tests.append(Y_test)

    return X_trains[index], X_tests[index], Y_trains[index], Y_tests[index]