import pandas as pd
from model import *

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from itertools import chain, repeat, islice

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

    #topic = pd.read_csv(topic_file,index_col=0)

    reddit_sample = news_table.sample(n=100, random_state=1)
    factcheck_sample = factcheck.sample(n=10, random_state=1)

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

    attention_masks, input_ids = vectorize(text_title_combined)  # tokenization + vectorization

    news_df = news_df.assign(attention_mask=attention_masks,
                                             input_id=input_ids.tolist())  # Assign mask and input_ids back to dataframe



    X_train, Y_train, X_val, Y_val, \
    train_inputs, train_masks, validation_inputs, validation_masks,\
    train_dataloader, validation_dataloader = vector_to_input(news_df,attention_masks,input_ids,mode='news')

    return X_train, Y_train, X_val, Y_val, \
           train_inputs, train_masks, validation_inputs, validation_masks,\
           train_dataloader, validation_dataloader

def get_comments_data():
    #_, comments_df, _ = import_data()
    comments_df = pd.read_csv('temp_comments_df.csv')

    print('Length of comments:', len(comments_df))

    attention_masks, input_ids = vectorize(comments_df.comment_text.values)
    comments_df.assign(attention_mask=attention_masks,
                       input_id=input_ids.tolist())


    X_train, Y_train, X_val, Y_val, \
    train_inputs, train_masks, validation_inputs, validation_masks,\
    train_dataloader, validation_dataloader = vector_to_input(comments_df,attention_masks,input_ids,mode='comments')

    return X_train, Y_train, X_val, Y_val, \
           train_inputs, train_masks, validation_inputs, validation_masks,\
           train_dataloader, validation_dataloader

# ========================================
#             Helper Functions
# ========================================
def vectorize(text,MAX_LEN=MAX_LEN):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    for t in text:
        # so basically encode tokenizing , mapping sentences to thier token ids after adding special tokens.
        encoded_sent = tokenizer.encode(
            t,  # Sentence which are encoding.
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

def single_word_vec(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_sent = tokenizer.encode(
        word,  # Sentence which are encoding.
        add_special_tokens=True,  # Adding special tokens '[CLS]' and '[SEP]'
    )

    encoded_sent = list(pad(encoded_sent,5,0))

    return encoded_sent

def vector_to_input(df,attention_masks,input_ids,mode):
    labels = df.label.values

    if mode=='news':
        # X index: input_id, topic, author
        temp = input_ids.tolist()
        author = []
        for a in df['author'].fillna(0):
            if a == 0:
                author.append([0,0,0,0,0])
            else:
                vec = single_word_vec(a)
                author.append(vec)
        for i in range(0,len(df)):
            temp[i].extend(author[i])
            temp[i].append(int(df['topic'][i]))
        X = np.array(temp)

    elif mode=='comments':
        # X index: input_id, comment_author, comment_score, comment_subreddit
        temp = input_ids.tolist()
        author = []
        for a in df['comment_author'].fillna(0):
            if a == 0:
                author.append([0,0,0,0,0])
            else:
                vec = single_word_vec(a)
                author.append(vec)
        subreddit = []
        for s in df['comment_subreddit'].fillna(0):
            if s == 0:
                subreddit.append([0, 0, 0, 0, 0])
            else:
                vec = single_word_vec(s)
                subreddit.append(vec)
        for i in range(0, len(df)):
            temp[i].extend(author[i])
            temp[i].extend(subreddit[i])
            temp[i].append(int(df['comment_score'][i]))
        X = np.array(temp)

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=1,
                                                                                        test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                           random_state=1, test_size=0.2)

    train_X, val_X, _, _ = train_test_split(X, labels, random_state=1, test_size=0.2)

    # changing the numpy arrays into tensors for working on GPU.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # DataLoader for our validation(test) set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_X, train_labels, val_X, validation_labels, \
           train_inputs, train_masks, validation_inputs, validation_masks,\
           train_dataloader, validation_dataloader


def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)