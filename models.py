import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class MRC_Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(MRC_Bert_BiLSTM_CRF, self).__init__()
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
    
    def _get_features(self, sentence, attention_mask, token_type_ids):
        with torch.no_grad():
          embeds, _ = self.bert(sentence, attention_mask=attention_mask, token_type_ids=token_type_ids)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, attention_mask, token_type_ids, is_test=False):
        emissions = self._get_features(sentence, attention_mask, token_type_ids)
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(emissions, tags, attention_mask > 0, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, attention_mask > 0)
            return decode
