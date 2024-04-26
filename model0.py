import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForTokenClassification

class Bert(torch.nn.Module):
    def __init__(self, tag_to_ix):
        super(Bert, self).__init__()
        self.tagset_size = len(tag_to_ix)
        self.bert = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=self.tagset_size)

    def forward(self, input_id, label, mask):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        # print(output)
        loss = output[0]
        logits = output[1]
        return loss, logits