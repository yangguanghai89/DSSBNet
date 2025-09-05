import torch
import random
import numpy as np
from datetime import datetime
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset

def get_parsere(parser):
    parser.add_argument('-train_path', type = str, default = '/home/zy/experiment/data/V6_t/train_index.tsv')
    parser.add_argument('-valid_path', type = str, default = '/home/zy/experiment/data/V6_t/dev_index.tsv')
    parser.add_argument('-test_path', type = str, default = '/home/zy/experiment/data/V6_t/test.tsv')
    parser.add_argument('-bert_path', type = str, default = '/home/zy/experiment/premodel/bert_base_uncase')
    # parser.add_argument('-ipc_path', type=str, default='/home/zy/secondly/data/V6_IPC/ipc.tsv')
    parser.add_argument('-batch_size', type = int, default = 32, help = 'input batch size for train and vaild')
    parser.add_argument('-learning_rate', type = float, default = 2e-5)
    parser.add_argument('-seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-epoch', type = int, default = 10, help = 'training rounds')
    parser.add_argument('-kernel_mul', type = float, default = 2.0, help='核函数的倍数因子')
    parser.add_argument('-kernel_num', type = int, default = 5, help='多尺度核函数的数量')
    parser.add_argument('-dropout', type = float, default = 0.1, help = 'the value of dropout')
    parser.add_argument('-alpha', type = float, default = 0.5, help = '样本权重正则化系数')
    parser.add_argument('-beta', type = float, default = 0.3, help='mmd加权系数')
    parser.add_argument('-device', type = str, default = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    return args

def setting(args, train_data, valid_data):
    print('训练集的大小是{}'.format(len(train_data)))
    print('验证集的大小是{}'.format(len(valid_data)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

def print_time(s):
    now = datetime.now()
    print(s, now)
    return now

def batch_data(args, data):
    text_a = data['text_a']
    text_b = data['text_b']
    bu = data['bu'].to(args.device)
    dalei = data['dalei'].to(args.device)
    xiaolei = data['xiaolei'].to(args.device)
    label = data['label'].to(args.device)
    index = data['index'].to(args.device)
    patentA = data['patentA']
    patentB = data['patentB']

    input_data = {
            'text_a': text_a,
            'text_b': text_b,
            'bu': bu,
            'patentA' : patentA,
            'patentB': patentB,
            'dalei': dalei,
            'xiaolei': xiaolei,
            'label': label,
            'index': index,
    }
    return input_data

class load_data_withopen(Dataset):
    def __init__(self, file_path, args):
        self.data = []
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        with open(file_path, 'r') as file:
            headers = file.readline().strip().split('\t')
            for line in file:
                parts = line.strip().split('\t')
                feature_dict = {}
                for i, header in enumerate(headers):
                    feature_dict[header] = parts[i]
                self.data.append(feature_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_dict = self.data[idx]
        text_a = feature_dict.get('#1 String')
        text_b = feature_dict.get('#2 String')
        patentA = feature_dict.get('#1 ID')
        patentB = feature_dict.get('#2 ID')
        bu = torch.tensor(float(feature_dict.get('ipc_1')))
        dalei = torch.tensor(float(feature_dict.get('ipc_2')))
        xiaolei = torch.tensor(float(feature_dict.get('ipc_3')))
        label = torch.tensor(float(feature_dict.get('Quality')))
        index = torch.tensor(int(feature_dict.get('index')))


        return {
            'text_a': text_a,
            'text_b': text_b,
            'bu': bu,
            'patentA' : patentA,
            'patentB': patentB,
            'dalei': dalei,
            'xiaolei': xiaolei,
            'label': label,
            'index': index,
        }