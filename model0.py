import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForTokenClassification
from transformers import BertPreTrainedModel, BertModel

class Bert(nn.Module):
    def __init__(self, tag_to_ix):
        super(Bert, self).__init__()
        self.tagset_size = len(tag_to_ix)
        self.bert = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=self.tagset_size)
        # self.bert = BertForTokenClassification.from_pretrained(
        #                'bert-base-cased', 
        #                              num_labels=self.tagset_size)
        self.linear = nn.Linear(self.embedding_dim, self.tagset_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_id, label, mask):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        output = self.dropout(output)
        # print(output)
        loss = output[0]
        logits = F.log_softmax(output[1], dim=2)
        return loss, logits
    

class BERT_CRF(BertPreTrainedModel):
    def __init__(self, tag_to_ix):
        super(BERT_CRF, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.dropout = nn.Dropout(p=0.1)
        # out_dim = config.hidden_size
        self.tagset_size = len(tag_to_ix)

        self.hidden2tag = nn.Linear(in_features=768, out_features=self.tagset_size)

        self.crf = CRF(num_tags=self.tagset_size, batch_first=True)

    def forward(self, inputs_ids, tags, token_type_ids=None, attention_mask=None):
        outputs = self.bert(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = output[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())