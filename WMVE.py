
from config import *
from preprocess import *
import torch
from random import randrange
from evaluation import *

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def WMVEpredict(weight:list, preds,use_softmax=False,final=False):
    softmax = torch.nn.Softmax()
    result = []
    prob = []
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
                prob.append((votes[1] / (votes[1] + votes[0])).item())
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
                prob.append((votes[1] / (votes[1] + votes[0])).item())
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
            prob.append((votes[1] / (votes[1] + votes[0])).item())
    return result, prob

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
            weight1[j] = weight1[j] / total_weight1

        total_weight2 = sum(weight2)
        for j in range(0, num+1):
            weight2[j] = weight2[j] / total_weight2

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
            weight[j] = weight[j] /total_weight
        return weight

def comments_voting(mode='test'):

    news_df = pd.read_csv('temp_news_df.csv')
    comments_df = pd.read_csv('temp_comments_df.csv')
    relationship = pd.read_csv('temp_relationship.csv')

    news_index = news_df.news_index.values
    labels = news_df.label.values

    over = SMOTE(sampling_strategy=0.4, random_state=1)
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=1)
    resample = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=resample)

    news_index, labels = pipeline.fit_resample(news_index.reshape(-1, 1), labels)

    result = []
    comment_result = []
    comment_label = []
    train_index, validation_index, train_label, val_label = k_fold_split(news_index, labels)
    train_index, test_index, train_label, test_label = train_test_split(train_index, train_label, random_state=1, test_size=0.2)

    has_comment_index = relationship.news_index.values

    track = 0
    if mode=='train':
        for i in train_index:
            if i in has_comment_index:
                comments = comments_df[comments_df['news_index'] == i[0]].reset_index(drop=True)
                commentvote = voting_for_one_news(comments, tokenizer)
                result.append(commentvote)
                comment_result.append(commentvote)
                comment_label.append(labels[track])
            else:
                result.append(2)
            track += 1
        binary_eval_comment('comment_train', comment_result,comment_label)
    elif mode=='test':
        for i in test_index:
            if i in has_comment_index:
                comments = comments_df[comments_df['news_index'] == i[0]].reset_index(drop=True)
                commentvote = voting_for_one_news(comments, tokenizer)
                result.append(commentvote)
                comment_result.append(commentvote)
                comment_label.append(labels[track])
            else:
                result.append(2)
            track += 1
        binary_eval_comment('comment_train', comment_result,comment_label)
    elif mode=='val':
        for i in validation_index:
            if i in has_comment_index:
                comments = comments_df[comments_df['news_index'] == i[0]].reset_index(drop=True)
                commentvote = voting_for_one_news(comments,tokenizer)
                result.append(commentvote)
                comment_result.append(commentvote)
                comment_label.append(labels[track])
            else:
                result.append(2)
            track += 1
        binary_eval_comment('comment_val', comment_result,comment_label)
    return result

def voting_for_one_news(df,tokenizer):
    attention_masks, input_ids = vectorize(df.comment_text.values,tokenizer, MAX_LEN=32)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    bert_c = torch.load(save_comment_model,map_location=device)

    bert_train_c = bertpredict(bert_c, input_ids.to(device), attention_masks.to(device))

    votes = [0,0]
    for i in bert_train_c:
        votes[i] += 1

        if votes[0] > votes[1]:
            return 0
        elif votes[0] < votes[1]:
            return 1
        else:
            return randrange(2)