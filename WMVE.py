
from config import *
from preprocess import *
import torch
from random import randrange

def WMVEpredict(weight:list, preds,use_softmax=False,final=False):
    softmax = torch.nn.Softmax()
    result = []
    bert_pre = preds['bert'].values
    forest_pre = preds['forest'].values
    nb_pre = preds['nb'].values
    lr_pre = preds['lr'].values
    if final:
        comment_pre = preds['comment'].values
        weight1, weight2 = weight
        for i in range(0, len(bert_pre)):
            if comment_pre[i] == 2:
                labels = torch.FloatTensor([0, 0])
                labels[bert_pre[i]] += weight1[0]
                labels[forest_pre[i]] += weight1[1]
                labels[nb_pre[i]] += weight1[2]
                labels[lr_pre[i]] += weight1[3]

                if use_softmax:
                    votes = softmax(labels)
                else:
                    votes = labels

                if votes[0] > votes[1]:
                    result.append(0)
                elif votes[0] < votes[1]:
                    result.append(1)
                else:
                    result.append(randrange(2))
            else:
                labels = torch.FloatTensor([0, 0])
                labels[bert_pre[i]] += weight2[0]
                labels[forest_pre[i]] += weight2[1]
                labels[nb_pre[i]] += weight2[2]
                labels[lr_pre[i]] += weight2[3]
                labels[comment_pre[i]] += weight2[4]

                if use_softmax:
                    votes = softmax(labels)
                else:
                    votes = labels

                if votes[0] > votes[1]:
                    result.append(0)
                elif votes[0] < votes[1]:
                    result.append(1)
                else:
                    result.append(randrange(2))
    else:
        weight = weight[0]
        for i in range(0, len(bert_pre)):
            labels = torch.FloatTensor([0,0])
            labels[bert_pre[i]] += weight[0]
            labels[forest_pre[i]] += weight[1]
            labels[nb_pre[i]] += weight[2]
            labels[lr_pre[i]] += weight[3]

            if use_softmax:
                votes = softmax(labels)
            else:
                votes = labels

            if votes[0] > votes[1]:
                result.append(0)
            elif votes[0] < votes[1]:
                result.append(1)
            else:
                result.append(randrange(2))

    return result

def train_weight(preds, label, num=4,final = False):
    bert_pre = preds['bert'].values
    forest_pre = preds['forest'].values
    nb_pre = preds['nb'].values
    lr_pre = preds['lr'].values
    if final:
        weight1 = [1]*num
        weight2 = [1]*(num+1)
        comment_pre = preds['comment'].values
        for i in range(0,len(label)):
            if comment_pre[i] == 2:
                update = [0]*num
                wrong = 0
                if bert_pre[i] == label[i]:
                    update[0] = 1
                else:
                    wrong += 1

                if forest_pre[i] == label[i]:
                    update[1] = 1
                else:
                    wrong += 1

                if nb_pre[i] == label[i]:
                    update[2] = 1
                else:
                    wrong += 1

                if lr_pre[i] == label[i]:
                    update[3] = 1
                else:
                    wrong += 1

                for j in range(0,num):
                    weight1[j] += update[j]* wrong/num
            else:
                update = [0]*(num+1)
                wrong = 0
                if bert_pre[i] == label[i]:
                    update[0] = 1
                else:
                    wrong += 1

                if forest_pre[i] == label[i]:
                    update[1] = 1
                else:
                    wrong += 1

                if nb_pre[i] == label[i]:
                    update[2] = 1
                else:
                    wrong += 1

                if lr_pre[i] == label[i]:
                    update[3] = 1
                else:
                    wrong += 1

                if comment_pre[i] == label[i]:
                    update[4] = 1
                else:
                    wrong += 1

                for j in range(0,num+1):
                    weight2[j] += update[j]* wrong/(num+1)

        total_weight1 = sum(weight1)
        for j in range(0, num):
            weight1[j] = j/total_weight1

        total_weight2 = sum(weight2)
        for j in range(0, num+1):
            weight2[j] = j / total_weight2

        return weight1, weight2

    else:
        weight = [1]*num
        for i in range(0,len(label)):
            update = [0]*num
            wrong = 0
            if bert_pre[i] == label[i]:
                update[0] = 1
            else:
                wrong += 1

            if forest_pre[i] == label[i]:
                update[1] = 1
            else:
                wrong += 1

            if nb_pre[i] == label[i]:
                update[2] = 1
            else:
                wrong += 1

            if lr_pre[i] == label[i]:
                update[3] = 1
            else:
                wrong += 1

            for j in range(0,num):
                weight[j] += update[j]* wrong/num

        total_weight = sum(weight)
        for j in range(0, num):
            weight[j] = j/total_weight
        return weight

def comments_voting(weight,mode='train'):
    #news_df, comments_df, relationship = import_data()

    news_df = pd.read_csv('temp_news_df.csv')
    comments_df = pd.read_csv('temp_comments_df.csv')
    relationship = pd.read_csv('temp_relationship.csv')

    news_index = news_df.news_index.values
    labels = news_df.label.values

    result = []
    train_index, validation_index, _, _ = train_test_split(news_index, labels, random_state=1, test_size=0.2)

    has_comment_index = relationship.news_index.values

    if mode=='train':
        for i in train_index:
            if i in has_comment_index:
                comments = comments_df[comments_df['news_index'] == i].reset_index(drop=True)
                result.append(voting_for_one_news(weight,comments))
            else:
                result.append(2)
    elif mode=='val':
        for i in validation_index:
            if i in has_comment_index:
                comments = comments_df[comments_df['news_index'] == i].reset_index(drop=True)
                result.append(voting_for_one_news(weight,comments))
            else:
                result.append(2)
    return result

def voting_for_one_news(weight,df):
    attention_masks, input_ids = vectorize(df.comment_text.values,MAX_LEN=32)

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

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    bert_c = torch.load(save_comment_model)
    forest_c = load(forest_comment_model)
    nb_c = load(nb_comment_model)
    lr_c = load(lr_comment_model)

    bert_train_c = bertpredict(bert_c, input_ids, attention_masks)
    forest_train_c= forest_c.predict(X)
    nb_train_c = nb_c.predict(X)
    lr_train_c = lr_c.predict(X)

    classfiers_pred_train_c = pd.DataFrame({'bert': bert_train_c,
                                    'forest': forest_train_c,
                                    'nb': nb_train_c,
                                    'lr': lr_train_c})

    voting_pred = WMVEpredict([weight], classfiers_pred_train_c)

    votes = [0,0]
    for i in voting_pred:
        votes[i] += 1

        if votes[0] > votes[1]:
            return 0
        elif votes[0] < votes[1]:
            return 1
        else:
            return randrange(2)