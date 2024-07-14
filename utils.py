import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
from models import MRC_Bert_BiLSTM_CRF
import copy
from torch.nn.utils.rnn import pad_sequence

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-BODY', 'I-TEST', 'I-EXAMINATIONS',
         'I-TREATMENT', 'B-DRUG', 'B-TREATMENT', 'I-DISEASES', 'B-EXAMINATIONS',
         'I-BODY', 'B-TEST', 'B-DISEASES', 'I-DRUG')
template = [("请找出句子中提及的药物，指用于疾病治疗的具体化学物质", "DRUG"),
            ("请找出句子中提及的解剖部位，指疾病、症状和体征发生的人体解剖学部位", "BODY"),
            ("请找出句子中提及的疾病和诊断，指医学上定义的疾病和医生在临床工作中对病因、病生理、分型分期等所作的判断", "DISEASES"),
            ("请找出句子中提及的检查，指影像检查（X线、CT、MR、PETCT等）+造影+超声+心电图", "EXAMINATIONS"),
            ("请找出句子中提及的检验，指在实验室进行的物理或化学检查", "TEST"),
            ("请找出句子中提及的手术，指医生在患者身体局部进行的切除、缝合等治疗，是外科的主要治疗方法", "TREATMENT")]
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256


class NerDataset(Dataset):
    ''' Generate our dataset '''
    def __init__(self, data_path) -> None:
        super().__init__()
        chars = []
        labels = []
        self.data_set = []
        with open(data_path, encoding='utf-8') as rf:
            for line in rf:
                line = line.split('\n')[0]
                char = line.split('\t')[0]
                label = line.split('\t')[1]
                if char != '。':
                    chars.append(char)
                    labels.append(label)
                else:
                    for predix, target in template:
                        # Predix是问题的前缀
                        # target是目标类型
                        # inputs_ids_1表示输入模板的token id
                        input_ids_1 = [tokenizer.convert_tokens_to_ids(c) for c in predix]
                        input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
                        # Bert两个句子进行拼接时候，token_type_ids:在第一个句子的位置为0，第二个句子为1，也就是不计入问题向量
                        token_type_ids_1 = [0] * len(input_ids_1)

                        # 长度裁剪到max_len以内
                        if len(chars) + 1 + len(input_ids_1) > MAX_LEN:
                            chars = chars[:MAX_LEN - 1 - len(input_ids_1)]
                            labels = labels[:MAX_LEN - 1 - len(input_ids_1)]
                        labels_ = copy.deepcopy(labels)

                        # inputs_ids_2表示数据集中每句话的token id
                        input_ids_2 = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                        input_ids_2 = input_ids_2 + [tokenizer.sep_token_id]
                        token_type_ids_2 = [1] * len(input_ids_2)
                        labels_ = labels_ + ["[SEP]"]

                        # 将与问题无关的实体都标记为O
                        for i in range(len(labels_)):
                            if labels_[i] == '<PAD>' or labels_[i] == '[CLS]' or labels_[i] == '[SEP]':
                                pass
                            elif target not in labels_[i]:
                                labels_[i] = 'O'
                            labels_[i] = tag2idx[labels_[i]]
                        labels_ids = [3] * len(input_ids_1) + labels_  # 问题前缀的标签无效

                        input_ids = input_ids_1 + input_ids_2
                        token_type_ids = token_type_ids_1 + token_type_ids_2
                        # print(len(labels_ids))
                        assert len(input_ids) == len(token_type_ids) == len(labels_ids)
                        self.data_set.append({"input_ids": input_ids,
                                              "token_type_ids": token_type_ids,
                                              "label_ids": labels_ids,
                                              "attention_mask": [1] * len(input_ids)})
                    chars = []
                    labels = []

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


def PadBatch(batch):
    input_ids_list, token_type_ids_list, label_ids_list, attention_mask_list = [], [], [], []
    for instance in batch:
        # 按照batch中的最大数据长度，对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        label_ids_temp = instance["label_ids"]
        attention_mask_temp = instance["attention_mask"]
        # 添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        label_ids_list.append(torch.tensor(label_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=1),
            "label_ids": pad_sequence(label_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels_name, title="Confusion Matrix",  is_norm=True,  colorbar=True, cmap=plt.cm.Blues):
    if is_norm == True:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)  # 横轴归一化并保留2位小数

    plt.figure(figsize=(12, 12))  # 调整图像大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')  # 默认所有值均为黑色
    if colorbar:
        plt.colorbar()  # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=60)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.title(title)  # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if is_norm==True:
        plt.savefig(r'cm_norm' + '.png', format='png', dpi=600)
    else:
        plt.savefig(r'cm' + '.png', format='png', dpi=600)
    plt.show()  # plt.show()在plt.savefig()之后


def inference(model, text, device, display=False):
    model.eval()
    token = tokenizer.tokenize(text)    # 将句子拆分为token

    # 生成模型输入，size为(batch_size=1, sequence_length)
    ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(token)]).to(device)   # 将token转换为数字
    mask = torch.BoolTensor([[True] * len(token)])
    mask = mask.to(device)
    with torch.no_grad():
        # 调用模型进行推断
        decode = model(ids, 0, mask, is_test=True)
    # 解码结果
    decoded_tags = [idx2tag[i] for i in decode[0]]

    # 输出推断结果
    flag = 0
    entities = {}
    label = ''
    position = [0, 0]
    if display:
        for i, tag in enumerate(decoded_tags):
            if tag[0] == 'B':
                flag = 1
                label = tag[2:]     # 实体类别
                if label not in entities.keys():
                    entities[label] = {}
                position[0] = i     # 实体位置
            elif tag[0] == 'I':
                position[1] = i
            elif flag == 1:
                entity = text[position[0]: position[1] + 1]     # 实体内容
                if entity not in entities[label].keys():
                    entities[label][entity] = [copy.deepcopy(position)]
                else:
                    entities[label][entity].append(copy.deepcopy(position))
                flag = 0
    print("text:{}".format(text))
    print("entity:{}".format(entities))

    result = {}
    for i in range(len(token)):
        result[token[i]] = decoded_tags[i]
    return result


if __name__ == "__main__":
    train_dataset = NerDataset("./data_all/processed_data/train_dataset.txt")
    train_dataloader = DataLoader(train_dataset, 32, shuffle=True, collate_fn=PadBatch, num_workers=4)
    for idx, data in enumerate(train_dataloader):
        print(data["label_ids"].shape)  # torch.Size([32, 256])
        print(data["label_ids"].view(-1).shape)  # torch.Size([32, 256])
        break
    print(len(train_dataset))
    print(train_dataset.__getitem__(0))
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # text = "患者4月前发现皮肤、巩膜黄染，伴食欲下降，晚餐后明显"
    # model = Bert_BiLSTM_CRF(tag2idx).cuda()
    # model.load_state_dict(torch.load("checkpoints/BERT_BiLSTM_CRF.pth"))
    # result = inference(model, text, device, display=True)
    # print(result)
