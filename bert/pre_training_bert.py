import os
import sys

sys.path.append(os.path.abspath('..'))
from config import Config

config = Config()

from torch import nn
import torch


class PreTrainingBert(nn.Module):
    def __init__(self, bert, vocab_size, hidden_size):
        super(PreTrainingBert, self).__init__()
        self.bert = bert
        self.mlm_linear = nn.Linear(hidden_size, vocab_size).to(config.device)
        self.nsp_linear = nn.Linear(hidden_size, 2).to(config.device)

    def _masked_lm(self, token):
        return self.mlm_linear(token).to(config.device)

    def _next_sentence_prediction(self, token):
        return self.nsp_linear(token[:, 0]).to(config.device)

    def pre_training(self, optimizer, criterion, data_loader):

        for epoch in range(config.epochs):
            for batch_data in data_loader:
                optimizer.zero_grad()
                data = {key: value.to(config.device) for key, value in batch_data.items()}
                bert_out = self.bert(data['token'], data['segment'])
                nsp_out = self.nsp_linear(bert_out[:, 0])
                mlm_out = self.mlm_linear(bert_out)
                nsp_loss = criterion(nsp_out, data['nsp'])
                mlm_loss = criterion(mlm_out.transpose(1, 2), data['mask'])
                loss = nsp_loss + mlm_loss
                loss.backward()
                lr = optimizer.step_and_update_learning_rate()
                print('epoch:', epoch, '损失率:', loss.detach(), '学习率：', lr)
            # 模型保存，只保存最后一次的训练结果
            '''保存整个模型'''
            if os.path.exists('model/bert_model.pth'):
                os.remove('model/bert_model.pth')
            torch.save(self.bert, 'model/bert_model.pth')
            '''保存模型参数'''
            if os.path.exists('model/bert_model_state.pkl'):
                os.remove('model/bert_model_state.pkl')
            torch.save(self.bert.state_dict(), 'model/bert_model_state.pkl')
