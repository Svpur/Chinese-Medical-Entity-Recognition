# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for building model. 
@All Right Reserve
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertForTokenClassification
from transformers import BertModel, BertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from torchcrf import CRF
from utils import tag2idx
from graphviz import Digraph

class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        with torch.no_grad():
          embeds, _  = self.bert(sentence)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        # print("feats:", feats)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        # print("tags:", tags)
        emissions = self._get_features(sentence)
        # print("emissions:", emissions)
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            # print("decode:", decode)
            return decode
        
        
class Bert(torch.nn.Module):
    def __init__(self, tag_to_ix, embedding_dim=768):
        super(Bert, self).__init__()
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.linear = nn.Linear(self.embedding_dim, self.tagset_size)

        self.dropout = nn.Dropout(p=0.1)
        self.loss=nn.CrossEntropyLoss()
        self.log_softmax = F.log_softmax

    def _get_features(self, sentence, mask):
        embeds, _  = self.bert(sentence, attention_mask=mask)
        # print("embeds:",embeds.shape)
        
        enc = self.dropout(embeds)
        # enc = torch.flatten(enc, start_dim=1)
        enc = self.linear(enc)
        # print("enc:",enc.shape)
        # feats = self.classifier(enc)
        return enc

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence, mask)
        print("emissions:",emissions)
        emissions = self.log_softmax(emissions, dim=-1)
        print("emissions_sm:",emissions)
        # print("emissions:",emissions.shape)
        # print("tags:",tags.shape)
        # print("transpose:",emissions.transpose(1,2).shape)
        if not is_test: # Training，return loss
            loss = self.loss(emissions.transpose(1,2),tags)  # emissions.transpose(1,2) -> batch_size*class_num*max_len
            return loss
        else: # Testing，return decoding
            # print("emissions:",emissions)
            # print("softmax:",emissions)
            # 在模型的 forward 方法中生成预测标签时使用 mask
            predicted_labels = torch.argmax(emissions, dim=2)
            # print("未填充前:",predicted_labels)
            # predicted_labels_masked = predicted_labels.masked_fill(~mask, -1)  # 将填充项设置为 -1
            # 将预测的标签张量转换为列表，并过滤掉填充项
            # predicted_labels_list = [label.tolist() for label in predicted_labels_masked] 
            # predicted_labels_list = [[lab for lab in seq if lab != -1] for seq in predicted_labels_masked]
            # print("predicted_labels_list:",len(predicted_labels_list))
            # print(predicted_labels_list)
            # print()
            return predicted_labels
        
class Bert_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768):
        super(Bert_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.embedding_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        embeds, _  = self.bert(sentence)
        enc = self.dropout(embeds)
        feats = self.linear(enc)
        # print("feats:", feats)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        # print("tags:", tags)
        emissions = self._get_features(sentence)
        # print("emissions:", emissions)
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            # print("decode:", decode)
            return decode

class AlBert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=1024, hidden_dim=256): # 768
        super(AlBert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # 使用ALBERT模型
        self.bert = BertModel.from_pretrained('voidful/albert_chinese_large', return_dict=False)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        with torch.no_grad():
            embeds, _  = self.bert(sentence)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        # print("feats:", feats)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        # print("tags:", tags)
        emissions = self._get_features(sentence)
        # print("emissions:", emissions)
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            # print("decode:", decode)
            return decode
        
if __name__=="__main__":
    model = Bert_BiLSTM_CRF(tag2idx)
    print(model)

    # 将模型转换为图形
    dot = model.create_graph(dot=True)
    # 生成图形
    with open("model_graph.dot", 'w') as f:
        f.write(dot)
    # 使用graphviz查看图形
    graph = Digraph()
    graph.from_source('model_graph.dot')
    graph.view()
