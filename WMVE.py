
from config import *
import torch
from random import randrange

def WMVEpredict(weight, preds,use_softmax=False):
    softmax = torch.nn.Softmax()
    result = []
    bert_pre = preds['bert'].values
    forest_pre = preds['forest'].values
    nb_pre = preds['nb'].values
    lr_pre = preds['lr'].values
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

def train_weight(preds, label, num=4):
    weight = [1]*num
    bert_pre = preds['bert'].values
    forest_pre = preds['forest'].values
    nb_pre = preds['nb'].values
    lr_pre = preds['lr'].values
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


