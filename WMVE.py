
from config import *
import torch

def WMVEpredict(weight, preds):
    result = []
    bert_pre = preds['bert'].values
    forest_pre = preds['forest'].values
    nb_pre = preds['nb'].values
    lr_pre = preds['lr'].values
    for i in range(0, len(bert_pre)):
        sum = weight[0]*bert_pre[i] + weight[1]*forest_pre[i] + weight[2]*nb_pre[i] + weight[3]*lr_pre[i]
        if sum >= cutoff:
            result.append(1)
        else:
            result.append(0)
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


