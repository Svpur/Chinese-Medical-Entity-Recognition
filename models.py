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
from transformers import BertForTokenClassification
from transformers import BertModel, BertTokenizer
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
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            return decode

# class Bert(torch.nn.Module):
#     def __init__(self, config, is_training, num_labels=11, dropout_prob=0.0):
#         super(Bert, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
#         # self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = lambda x, dtype: x.to(dtype)
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.log_softmax = F.log_softmax
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#         self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#         self.dropout = nn.Dropout(1 - dropout_prob)
#         self.reshape = lambda x, shape: x.view(shape)
#         self.shape = (-1, config.hidden_size)
#         self.origin_shape = (config.batch_size, config.seq_length, self.num_labels)
#         self.loss = CrossEntropyCalculation(is_training)
#         self.num_labels = num_labels

#     def forward(self, input_ids, label_ids, input_mask, is_test=False):
#         sequence_output, _, _ = \
#             self.bert(input_ids, input_mask)
#         seq = self.dropout(sequence_output)
#         seq = self.reshape(seq, self.shape)
#         logits = self.dense_1(seq)
#         logits = self.cast(logits, self.dtype)
            
#         if not is_test:
#             return_value = self.log_softmax(logits)
#             loss = self.loss(return_value, label_ids, self.num_labels)
#             return loss
#         else:
#             return logits
        
        
class Bert(torch.nn.Module):
    def __init__(self, tag_to_ix, embedding_dim=768):
        super(Bert, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.linear = nn.Linear(self.embedding_dim, self.tagset_size)
        self.dropout = nn.Dropout(p=0.1)
        self.loss=nn.CrossEntropyLoss()

    def _get_features(self, sentence, mask):
        embeds, _  = self.bert(sentence, attention_mask=mask)
        print("embeds:",embeds.shape)
        enc = self.linear(embeds)
        print("enc:",enc.shape)
        enc = self.dropout(enc)
        # enc = torch.flatten(enc, start_dim=1)
        
        # feats = self.classifier(enc)
        return enc

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence, mask)
        A = emissions.transpose(1,2)
        print("tags:",tags.shape)
        print("A:",A.shape)
        if not is_test: # Training，return loss
            loss = self.loss(emissions.transpose(1,2),tags)  # emissions.transpose(1,2) -> batch_size*class_num*max_len
            return loss
        else: # Testing，return decoding
            return emissions.transpose(1,2)
        
        
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
