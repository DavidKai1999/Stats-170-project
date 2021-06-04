
import pandas as pd
from sqlalchemy import create_engine
from train_bert import bertpredict_withbatch
from WMVE import *
from preprocess import *
from config import tokenizer
import pickle

from sklearn import preprocessing

def main(news_file, comments_file,relationship_file):
    '''
    news_df = pd.read_csv(news_file)
    comments_df = pd.read_csv(comments_file)
    relationship = pd.read_csv(relationship_file)

    X, Y, dataloader = embedding_news(news_df)


    bertnews = torch.load(save_news_model, map_location=device)
    bert_pred = bertpredict_withbatch(dataloader, bertnews)

    with open(".\\tempfile\\bert_pred.txt", "wb") as fp:  # Pickling
        pickle.dump(bert_pred, fp)

    comment_pred = comments_voting_pred(news_df, comments_df, relationship)

    with open(".\\tempfile\\comment_pred.txt", "wb") as fp:  # Pickling
        pickle.dump(comment_pred, fp)
    '''

    with open(".\\tempfile\\pred_data.txt", "rb") as fp:  # Unpickling
        X, Y = pickle.load(fp)
    with open(".\\tempfile\\bert_pred.txt", "rb") as fp:  # Unpickling
        bert_pred = pickle.load(fp)
    with open(".\\tempfile\\comment_pred.txt", "rb") as fp:  # Pickling
        comment_pred = pickle.load(fp)

    with open(".\\tempfile\\voting_weight.txt", "rb") as fp:  # Unpickling
        weight = pickle.load(fp)  # Load the voting weights

    scaler = preprocessing.StandardScaler().fit(X)
    X_scale = scaler.transform(X)

    # Load classifiers
    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    forest_pred = forest.predict(X)
    nb_pred = nb.predict(X_scale)
    lr_pred = lr.predict(X_scale)

    binary_eval('bert', bert_pred, Y)
    binary_eval('forest', forest_pred, Y)
    binary_eval('nb', nb_pred, Y)
    binary_eval('lr', lr_pred, Y)

    # Combine all predictions to feed to the weighted voting model
    classfiers_pred = pd.DataFrame({'bert': bert_pred,
                                    'forest': forest_pred,
                                    'nb': nb_pred,
                                    'lr': lr_pred,
                                    'comment': comment_pred})

    voting_pred, prob = WMVEpredict(weight, classfiers_pred, use_softmax=False, final=True)

    binary_eval('voting_val', Y, voting_pred)

def embedding_news(news_df):
    news_df['text_combined'] = news_df['text'] + news_df['title'] * 2
    text_title_combined = news_df.text_combined.values

    attention_masks, input_ids = vectorize(text_title_combined, tokenizer=tokenizer)  # tokenization + vectorization

    news_df = news_df.assign(attention_mask=attention_masks,
                             input_id=input_ids.tolist())  # Assign mask and input_ids back to dataframe

    X, Y, dataloader = vector_to_input_pred(news_df, attention_masks, input_ids,
                                            mode = 'news', tokenizer=tokenizer)

    return X, Y, dataloader

def embedding_comments(comments_df):
    attention_masks, input_ids = vectorize(comments_df.comment_text.values, tokenizer=tokenizer, MAX_LEN=20)
    comments_df.assign(attention_mask=attention_masks,
                       input_id=input_ids.tolist())

    X, Y, dataloader = vector_to_input_pred(news_df, attention_masks, input_ids,
                                            mode = 'comments', tokenizer=tokenizer)

    return X, Y, dataloader

def import_data():
    user = 'postgres'
    password = 'Komaeda'

    engine = create_engine('postgresql://'+user+':'+password+'@localhost/news')

    Query = "SELECT * FROM redditcomment"
    comment_table = pd.read_sql_query(Query, con=engine)
    Query = "SELECT title,text,label FROM redditnews"
    news_table = pd.read_sql_query(Query, con=engine)
    Query = "SELECT title,text,author,label FROM factcheck"
    factcheck = pd.read_sql_query(Query, con=engine)
    Query = "SELECT * FROM topic"
    topic = pd.read_sql_query(Query, con=engine)

    reddit_sample = news_table.sample(n=500, random_state=1)
    factcheck_sample = factcheck.sample(n=50, random_state=1)

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

    news_df.to_csv('pred_news.csv')
    comments_df.to_csv('pred_comments.csv')
    news_comments_relationship.to_csv('pred_relationship.csv')

def vector_to_input_pred(df,attention_masks,input_ids,mode,tokenizer):
    labels = df.label.values

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

        # changing the numpy arrays into tensors for working on GPU.
        inputs = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        masks = torch.tensor(attention_masks)

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

        # changing the numpy arrays into tensors for working on GPU.
        inputs = torch.tensor(X)
        labels = torch.tensor(labels)
        masks = torch.tensor(X_mask)

    # DataLoader for our training set.
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return X, labels, dataloader

def comments_voting_pred(news_df, comments_df, relationship):

    news_index = news_df.news_index.values.reshape(-1, 1)
    labels = news_df.label.values

    result = []
    comment_result = []
    comment_label = []

    has_comment_index = relationship.news_index.values

    track = 0
    for i in news_index:
        if i in has_comment_index:
            comments = comments_df[comments_df['news_index'] == i[0]].reset_index(drop=True)
            commentvote = voting_for_one_news(comments, tokenizer)
            result.append(commentvote)
            comment_result.append(commentvote)
            comment_label.append(labels[track])
        else:
            result.append(2)
        track += 1
    binary_eval_comment('comment_voting', comment_result, comment_label)
    return result

if __name__ == '__main__':
    #import_data()
    main('pred_news.csv','pred_comments.csv','pred_relationship.csv')