import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class net(nn.Module):
    def __init__(self,args):
        super(net,self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.bert_shared = BertModel.from_pretrained(args.bert_path)
        self.args = args

        self.REP_C = nn.Sequential(*self.rep_layer(input_dims = 768, out_dims = 768, layer = 3))
        self.REP_A = nn.Sequential(*self.rep_layer(input_dims=768, out_dims=768, layer=3))
        self.t_regress = nn.Sequential(*self.output_layer(input_dims=768, out_dims=2, layer=5))
        self.y_regress_0 = nn.Sequential(*self.output_layer(input_dims = 768 + 768 +64, out_dims = 2, layer = 5))
        self.y_regress_1 = nn.Sequential(*self.output_layer(input_dims=768 + 768 + 64, out_dims=2, layer=5))
        self.y_regress_a = nn.Sequential(*self.output_layer(input_dims=768, out_dims=2, layer=5))
        self.map_t = nn.Sequential(*self.rep_layer(input_dims=1, out_dims=64, layer=2))
        self.sample_weight = nn.Parameter(torch.ones(9952, 1))

    def forward(self, input_data):
        text_emb = self.bert(input_data['text_a'], input_data['text_b'])
        rep_t = self.map_t(input_data['xiaolei'].unsqueeze(1))
        rep_c = self.REP_C(text_emb)
        rep_a = self.REP_A(text_emb)

        pred_y_0 = self.y_regress_0(torch.cat((rep_c, rep_a, rep_t), dim = 1))
        pred_y_1 = self.y_regress_1(torch.cat((rep_c, rep_a, rep_t), dim=1))
        pred_y = (((1 - input_data['xiaolei']).unsqueeze(1) * pred_y_0 + input_data['xiaolei'].unsqueeze(1) * pred_y_1))[:, 1]
        a_pred_y = self.y_regress_a(rep_a)[:, 1]
        pred_t = self.t_regress(rep_c)[:, 1]
        output_data = {
            'text_emb' : text_emb,
            'rep_c' : rep_c,
            'rep_a' : rep_a,
            'a_pred_y' : a_pred_y,
            'pred_y' : pred_y,
            'pred_t' : pred_t
        }
        self.i0 = [i for i, value in enumerate(input_data['xiaolei']) if value == 0]
        self.i1 = [i for i, value in enumerate(input_data['xiaolei']) if value == 1]
        self.output = output_data
        self.input = input_data
        return output_data

    def bert(self, text_a, text_b):
        encode_text = self.tokenizer(text_a, text_b, add_special_tokens=True, return_tensors='pt', padding='max_length',
                                     truncation = True, max_length = 128)
        encoded_dict = encode_text.to(self.args.device)
        outputs = self.bert_shared(**encoded_dict)
        result = outputs.last_hidden_state[:, 0, :]

        return result

    def ipm(self):
        ipm = 0
        if self.input['tr']:
            if self.i0 and self.i1:
                c = self.sample_weight[self.input['index']] * self.output['rep_c']
                mean_c_0 = c[self.i0].mean(dim=0).unsqueeze(0)
                mean_c_1 = c[self.i1].mean(dim=0).unsqueeze(0)
                ipm = 1 - F.cosine_similarity(mean_c_0, mean_c_1)
        else:
            if self.i0 and self.i1:
                mean_c_0 = self.output['text_emb'][self.i0].mean(dim=0).unsqueeze(0)
                mean_c_1 = self.output['text_emb'][self.i1].mean(dim=0).unsqueeze(0)
                ipm = 1 - F.cosine_similarity(mean_c_0, mean_c_1)

        return ipm.abs()

    def loss_func(self):

        if self.input['tr']:
            a_loss_y = F.binary_cross_entropy(input=self.output['a_pred_y'], target=self.input['label'], reduction='mean')
            loss_y = F.binary_cross_entropy(input=self.output['pred_y'], target=self.input['label'], reduction='mean')
            loss_t = F.binary_cross_entropy(input=self.output['pred_t'], target=self.input['xiaolei'], reduction='mean')
            loss_reg = F.cosine_similarity(self.output['rep_c'], self.output['rep_a'], dim=1).abs().mean()
            ipm = self.ipm()
        else:
            a_loss_y = F.binary_cross_entropy(input=self.output['a_pred_y'], target=self.input['label'],reduction='mean')
            loss_y = F.binary_cross_entropy(input=self.output['pred_y'], target=self.input['label'], reduction='mean')
            loss_t = F.binary_cross_entropy(input=self.output['pred_t'], target=self.input['xiaolei'], reduction='mean')
            loss_reg = F.cosine_similarity(self.output['rep_c'], self.output['rep_a'], dim=1).abs().mean()
            ipm = self.ipm()

        _ = {'loss_y' : loss_y,
             'loss_t' : loss_t,
             'a_loss_y' : a_loss_y,
             'loss_reg' : loss_reg
        }
        loss = loss_t + loss_y + a_loss_y + loss_reg + ipm
        return loss, _

    def rep_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []

        for i in range(0, layer):
            layers.append(nn.Linear(dim[i], dim[i+1]))
            # if self.args.batch_norm:
            #     layers.append(nn.BatchNorm1d(dim[i+1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p = self.args.dropout))
        return layers

    def output_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []

        for i in range(layer):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            if i < layer - 1:
                # if self.args.batch_norm:
                #     layers.append(nn.BatchNorm1d(dim[i + 1]))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(p=self.args.dropout))
        layers.append(nn.Softmax())
        return layers