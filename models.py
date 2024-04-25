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


# class Bert(nn.Module):

#     def __init__(self, tag_to_ix, hidden_dim=768):
#         super(Bert, self).__init__()
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)
#         self.hidden_dim = hidden_dim

#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#         self.dropout = nn.Dropout(p=0.1)
#         self.linear = nn.Linear(hidden_dim, self.tagset_size)
    
#     def _get_features(self, input_ids, attention_mask):
#         _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
#         features = self.dropout(pooled_output)
#         return features

#     def forward(self, input_ids, attention_mask, tags=None, is_test=False):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         # 使用BERT的token-level输出，而非pooled_output
#         sequence_output = outputs[0]
        
#         # 应用dropout
#         sequence_output = self.dropout(sequence_output)
        
#         # 线性层应用于每个token的输出
#         logits = self.linear(sequence_output)

#         # 如果是训练模式
#         if not is_test:
#             # 确保tags中padding位置使用了ignore_index（0）
#             loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # padding标签为0
#             # 直接计算损失，CrossEntropyLoss内部会处理padding
#             loss = loss_fct(logits.view(-1, self.tagset_size), tags.view(-1).long())
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
        self.classifier=nn.Linear(self.embedding_dim,self.tagset_size)     
        self.dropout = nn.Dropout(p=0.1)
        self.loss=nn.CrossEntropyLoss()
    
    def _get_features(self, sentence, mask):
        embeds, _  = self.bert(sentence, attention_mask=mask)
        enc = self.dropout(embeds)
        # enc = torch.flatten(enc, start_dim=1)
        feats = self.classifier(enc[0])
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence, mask)
        if not is_test: # Training，return loss
            loss = self.loss(emissions.transpose(1,2),tags)  # emissions.transpose(1,2) -> batch_size*class_num*max_len
            return loss
        else: # Testing，return decoding
            return emissions

# class Bert(nn.Module):
#     def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):  # hidden_dim is not used for BertOnly
#         super(Bert, self).__init__()
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)
#         self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
#         # Since we are not using a linear layer transformation, the hidden_dim parameter is not used
        
#     def _get_features(self, sentence):
#         # We don't need to use no_grad context as we are not computing gradients in the forward pass anyway
#         embeds, _ = self.bert(sentence)
#         return embeds  # Directly return the embeddings from BERT
        
#     def forward(self, sentence, tags=None, mask=None, is_test=False):
#         features = self._get_features(sentence)
#         if not is_test:
#             # Define a linear layer here which was missing in the original code
#             logits = nn.functional.linear(features, self.tagset_size)
#             loss_fn = nn.CrossEntropyLoss(ignore_index=self.tag_to_ix.get('PAD', -100))  # Assuming 'PAD' is a valid tag and should be ignored
#             loss = loss_fn(logits.view(-1, self.tagset_size), tags.view(-1))
#             return loss
#         else:
#             # During testing, you might want to apply some kind of decoding strategy to the raw BERT outputs
#             # For example, using argmax to get the most probable tag for each token
#             probs = nn.functional.softmax(features, dim=2)
#             predictions = probs.argmax(dim=2)
#             return predictions
        
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
