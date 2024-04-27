# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for training, validating and testing. 
@All Right Reserve
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from model0 import Bert
from models import Bert_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        # print(step, "------------------------------------------")
        step += 1
        x, y, z = batch
        x = x.to(device) # input_id
        y = y.to(device) # train_label
        z = z.to(device) # mask

        loss, _ = model(x,y,z)

        # # 过滤掉特殊token及padding的token
        # logits_clean = logits[0][train_label != 0]
        # label_clean = train_label[train_label != 0]
        # # 获取最大概率值
        # predictions = logits_clean.argmax(dim=1)

        losses += loss.item()
        """ Gradient Accumulation """
        '''
          full_loss = loss / 2                            # normalize loss 
          full_loss.backward()                            # backward and accumulate gradient
          if step % 2 == 0:             
              optimizer.step()                            # update optimizer
              scheduler.step()                            # update scheduler
              optimizer.zero_grad()                       # clear gradient
        '''
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses/step))
    
def validate(e, model, iterator, device):
    model.eval()
    losses = 0
    total_acc_val = 0
    step = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1

            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            loss, logits = model(x,y,z)
            print("y:", y.shape)
            print("z:", z.shape)
            print("logits:", logits.shape)
            # print("logits[0]:", logits[0].shape)
            # print("logits[1]:", logits[1].shape)
            # print("logits[2]:", logits[2].shape)
            # 过滤掉特殊token及padding的token
            # 获取最大概率值
            logits_max = logits.argmax(dim=2)
            print("logits_max:", logits_max.shape)
            logits_clean = logits_max[z == 1]
            print("logits_clean:", logits_clean.shape)
            # M = logits_max[z]
            # print("M:", M.shape)

            label_clean = y[z == 1]
            print("label_clean:", label_clean.shape)
    
            predictions = logits_clean

            losses += loss.item()

            # 计算准确率
            print("predictions:", predictions)
            print("label_clean:", label_clean)
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, total_acc_val/step))
    return model, losses/step, total_acc_val/step


def test(model, iterator, device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            _, logits = model(x,y,z)
            # 过滤掉特殊token及padding的token
            logits_max = logits.argmax(dim=2)
            logits_clean = logits_max[y != 0]
            label_clean = y[y != 0]
            # 获取最大概率值
            predictions = logits_clean

    return predictions, label_clean


if __name__=="__main__":

    labels = ['B-BODY',
      'B-DISEASES',
      'B-DRUG',
      'B-EXAMINATIONS',
      'B-TEST',
      'B-TREATMENT',
      'I-BODY',
      'I-DISEASES',
      'I-DRUG',
      'I-EXAMINATIONS',
      'I-TEST',
      'I-TREATMENT']
    
    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--trainset", type=str, default="./CCKS_2019_Task1/processed_data/train_dataset.txt")
    parser.add_argument("--validset", type=str, default="./CCKS_2019_Task1/processed_data/val_dataset.txt")
    parser.add_argument("--testset", type=str, default="./CCKS_2019_Task1/processed_data/test_dataset.txt")
    parser.add_argument("--Model", type=str, default="Bert")

    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ner.Model == 'Bert_BiLSTM_CRF':
        print('Loading Bert_BiLSTM_CRF model.')
        model = Bert_BiLSTM_CRF(tag2idx).cuda()
    # elif ner.Model == 'Bert_CRF':
    #     print('Loading Bert_CRF model.')
    #     model = BiLSTM_CRF(tag2idx).cuda()
    elif ner.Model == 'Bert':
        print('Loading Bert model.')
        model = Bert(tag2idx).cuda()

    print('Initial model Done.')
    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    #optimizer = optim.Adam(self.model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset) 
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
    
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    print('Start Train...,')
    for epoch in range(1, ner.n_epochs+1):

        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)

        if loss < _best_val_loss and acc > _best_val_acc:
          best_model = candidate_model
          _best_val_loss = loss
          _best_val_acc = acc

        print("=============================================")
    
    y_test, y_pred = test(best_model, test_iter, device)
    print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))

    # 保存模型
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(best_model.state_dict(), "checkpoints/BERT_BiLSTM_CRF.pth")
    print("模型已保存！")
