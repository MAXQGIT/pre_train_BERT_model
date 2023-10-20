import os
import sys
from torch import nn
import numpy as np

sys.path.append(os.path.abspath('..'))
from config import Config
import torch
import numpy as np

config = Config()


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / np.power(config.d_k, 0.5)
        attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = torch.bmm(attn, v)
        return attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_ff, hidden_size):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(in_channels=hidden_size, out_channels=d_ff, kernel_size=1)
        self.w_2 = nn.Conv1d(in_channels=d_ff, out_channels=hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        ffn = self.w_1(x.transpose(1, 2))
        # 这个地方激活函数在utails中有gelu
        ffn = self.relu(ffn)
        ffn = self.w_2(ffn)
        ffn = self.dropout(ffn)
        ffn = self.layer_norm(residual + ffn.transpose(1, 2))
        return ffn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.w_q_s = nn.Linear(config.hidden_size, config.heads * config.d_q)
        self.w_k_s = nn.Linear(config.hidden_size, config.heads * config.d_k)
        self.w_v_s = nn.Linear(config.hidden_size, config.heads * config.d_v)
        nn.init.normal_(self.w_q_s.weight, mean=0, std=np.sqrt(2.0 / (config.hidden_size + config.d_q)))
        nn.init.normal_(self.w_k_s.weight, mean=0, std=np.sqrt(2.0 / (config.hidden_size + config.d_k)))
        nn.init.normal_(self.w_v_s.weight, mean=0, std=np.sqrt(2.0 / config.hidden_size + config.d_v))
        self.concat_heads = nn.Linear(config.heads * config.d_v, config.hidden_size)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, query, key, value, mask):
        residual = query
        batch_size, seq_len_q, dim = query.shape
        batch_size, seq_len_k, dim = key.shape
        batch_size, seq_len_v, dim = value.shape
        query = self.w_q_s(query).view(batch_size, seq_len_q,config.heads, config.d_q)
        key = self.w_k_s(key).view(batch_size, seq_len_k, config.heads, config.d_k)
        value = self.w_v_s(value).view(batch_size, seq_len_v, config.heads, config.d_v)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, config.d_q)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, config.d_k)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_v, config.d_v)
        pad_mask = mask.repeat(config.heads, 1, 1)
        attn = self.attention(query, key, value, pad_mask)
        attn = attn.view(config.heads, batch_size, seq_len_q, config.d_q)
        attn = attn.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len_q, -1)
        attn = self.dropout(self.concat_heads(attn))
        attn = self.layer_norm(residual + attn)
        return attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_ff, hidden_size):
        super(TransformerEncoderLayer, self).__init__()
        self.mutli_heads_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward(d_ff, hidden_size)

    def forward(self, token, pad_mask, no_pad_mask):
        attn = self.mutli_heads_attn(token, token, token, pad_mask)
        # 这个地方还需要仔细思考一下
        attn *= no_pad_mask
        output = self.feed_forward(attn)
        output *= no_pad_mask
        return output
