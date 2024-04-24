# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for building model. 
@All Right Reserve
'''
from transformers import BertModel

from torchcrf import CRF

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# 假设CRF是一个有效的条件随机场实现，你需要根据你的项目来定义或导入

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256, useBERT=True, useBiLSTM=True, useCRF=True):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.useBERT = useBERT
        self.useBiLSTM = useBiLSTM
        self.useCRF = useCRF

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False) if self.useBERT else None
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True) if self.useBiLSTM else None
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True) if self.useCRF else None

    def _get_features(self, sentence):
        # sentence_tensors: [batch_size, seq_len]
        # sentence_masks: [batch_size, seq_len]
        with torch.no_grad():
            if self.useBERT:
                embeds, _ = self.bert(sentence)
            else:
                embeds = sentence  
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:
            if self.useCRF:
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                loss = torch.nn.functional.cross_entropy(emissions.view(-1, self.tagset_size), tags.view(-1), reduction='mean')
            return loss
        else:
            if self.useCRF:
                decode=self.crf.decode(emissions, mask)
            else:
                decode = emissions.argmax(dim=-1)
            return decode
