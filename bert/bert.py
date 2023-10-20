import os
import sys
import numpy as np
import torch
from .transformer import TransformerEncoderLayer

sys.path.append(os.path.abspath('..'))
from config import Config
from .utils import Mask

config = Config()
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super(PositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        pos_enc = self.position_weight()
        self.pos_embedding = nn.Embedding(self.vocab_size, config.hidden_size)
        self.pos_embedding.from_pretrained(torch.FloatTensor(pos_enc))

    def position_weight(self):
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (i // 2) / config.hidden_size) for i in range(config.hidden_size)]
             for pos in range(self.vocab_size)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 0::2] = np.cos(pos_enc[:, 1::2])
        pos_enc[config.pad_idx] = 0.
        return pos_enc

    def forward(self, token):
        return self.pos_embedding(token.to(device=config.device))


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BertEmbedding, self).__init__()
        segment_num = 3  # 句子标记的
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=config.pad_idx)
        self.segment_embedding = nn.Embedding(segment_num, hidden_size)
        self.position_embedding = PositionEmbedding(vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token, segment):
        token_emb = self.token_embedding(token)
        pos_emb = self.position_embedding(token)
        seg_emb = self.segment_embedding(segment)
        bert_input = token_emb + pos_emb + seg_emb
        bert_input = self.dropout(bert_input)
        return bert_input


class Bert(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(Bert, self).__init__()
        self.embdedding = BertEmbedding(vocab_size, hidden_size)
        d_ff = hidden_size * 4
        self.transformer_block = nn.ModuleList(
            [TransformerEncoderLayer(d_ff, hidden_size) for _ in range(config.layers)]
        )
        self.mask_obj = Mask()

    def forward(self, token, segment):
        input_embedding = self.embdedding(token, segment)
        pad_mask = self.mask_obj.padding_mask(token, token)
        no_pad_mask = self.mask_obj.no_padding_mask(token)
        enc_out = None
        for layer in self.transformer_block:
            enc_out = layer(input_embedding, pad_mask, no_pad_mask)
        return enc_out
