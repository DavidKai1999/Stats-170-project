
from train_bert import bertpredict_withbatch
from WMVE import *
from preprocess import *
from config import tokenizer
import pickle

from sklearn import preprocessing

def main():
    news_df = pd.read_csv('pred_news.csv')
    comments_df = pd.read_csv('pred_comments.csv')
    relationship = pd.read_csv('pred_relationship.csv')

    print('Embedding context...')
    X, Y, dataloader = embedding_news(news_df)

    bertnews = torch.load(save_news_model, map_location=device)
    bert_pred = bertpredict_withbatch(dataloader, bertnews)

    comment_pred = comments_voting_pred(news_df, comments_df, relationship)

    with open(".\\tempfile\\voting_weight.txt", "rb") as fp:  # Unpickling
        weight = pickle.load(fp)  # Load the voting weights

    scaler = preprocessing.StandardScaler().fit(X)
    X_scale = scaler.transform(X)

    # Load classifiers
    forest = load(forest_news_model)
    nb = load(nb_news_model)
    lr = load(lr_news_model)

    print('Predicting...')

    forest_pred = forest.predict(X)
    nb_pred = nb.predict(X_scale)
    lr_pred = lr.predict(X_scale)

    # Combine all predictions to feed to the weighted voting model
    classfiers_pred = pd.DataFrame({'bert': bert_pred,
                                    'forest': forest_pred,
                                    'nb': nb_pred,
                                    'lr': lr_pred,
                                    'comment': comment_pred})

    print('Voting...')
    voting_pred, _ = WMVEpredict(weight, classfiers_pred, use_softmax=False, final=True)

    return voting_pred

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

def import_data(news_file, comment_file):

    news_table = pd.read_csv(news_file)

    if 'label' not in news_table.columns:
        news_table = news_table.assign(label = [0 for i in range(0,len(news_table))])

    news_df = news_table.assign(news_index = [i for i in range(0,len(news_table))])

    comment_table = pd.read_csv(comment_file)

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
    binary_eval_comment('comment_voting', comment_label,comment_result)
    return result

if __name__ == '__main__':
    # Import the news table and comment table 
    import_data('news_table.csv', 'comment_table.csv')
    voting_pred = main()
    print(voting_pred)