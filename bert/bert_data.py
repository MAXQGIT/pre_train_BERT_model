import os
import sys

import torch

sys.path.append(os.path.abspath('..'))
from config import Config
import random

config = Config()


class BertData:

    def __init__(self, data_path, vocab_obj):
        self.vocab_obj = vocab_obj
        with open(data_path, 'r', encoding='utf-8') as file:
            self.content = [line.strip('\n').split('\t') for line in file]
            self.lines = len(self.content)

    def build_mlm_nsp(self):

        def masked_sentence_words(sent, sent_a=None):
            sentence_token2mask = []
            sentence_token2idx = []
            '''英文数据这么处理，中文数据这里需要改动，删除split()'''
            for idx, token in enumerate(sent):
                token_mask_p = random.random()
                if token_mask_p < 0.15:
                    token_mask_type = token_mask_p / 0.15
                    if token_mask_type < 0.8:
                        sentence_token2idx.append(config.mask_idx)
                    elif token_mask_type < 0.9:
                        replace_token = random.choice(self.vocab_obj.words)
                        sentence_token2idx.append(self.vocab_obj.word2index[replace_token])
                    else:
                        sentence_token2idx.append(self.vocab_obj.word2index[token])
                    sentence_token2mask.append(self.vocab_obj.word2index[token])
                else:
                    sentence_token2idx.append(self.vocab_obj.word2index[token])
                    sentence_token2mask.append(config.no_mask)
            sentence_token2idx += [config.eos_idx]
            sentence_token2mask += [config.pad_idx]
            if sent_a:
                sentence_token2idx = [config.sos_idx] + sentence_token2idx
                sentence_token2mask = [config.pad_idx] + sentence_token2mask
            return sentence_token2mask, sentence_token2idx

        sent_a_b = []
        for line in self.content:
            sent_a = line[0]
            if random.random() < 0.5:
                sent_b = line[1]
                sent_a_b.append([sent_a, sent_b, config.IsNext])
            else:
                sent_b = self.content[random.randrange(len(self.content))][1]
                sent_a_b.append([sent_a, sent_b, config.NoNext])
        token_mask_segment_nsp = []
        for idx, line in enumerate(sent_a_b):
            token2mask_a, token2idx_a = masked_sentence_words(line[0], True)
            token2mask_b, token2idx_b = masked_sentence_words(line[1])
            segment_idx = ([1 for _ in range(len(token2idx_a))] + [2 for _ in range(len(token2idx_b))])[:config.max_predictions_per_seq]
            token_len = len(token2idx_a + token2idx_b)
            mask_len = len(token2mask_a + token2mask_b)
            assert token_len == mask_len
            bert_token = (token2idx_a + token2idx_b)[:config.max_predictions_per_seq]
            bert_mask = (token2mask_a + token2mask_b)[:config.max_predictions_per_seq]
            bert_token = bert_token + [config.pad_idx for _ in range(config.max_predictions_per_seq - token_len)]
            bert_mask = bert_mask + [config.pad_idx for _ in range(config.max_predictions_per_seq - mask_len)]
            bert_segment = segment_idx + [config.pad_idx for _ in
                                          range(config.max_predictions_per_seq - len(segment_idx))]
            bert_nsp = line[2]
            bert_mlm_nsp = {
                'token': bert_token,  # bert模型输入的Ttoken索引
                'mask': bert_mask,  # bert模型做maskmlm的mask索引
                'segment': bert_segment,  # bert模型的seqment索引
                'nsp': bert_nsp
            }
            bert_mlm_nsp = {key: torch.tensor(value) for key, value in bert_mlm_nsp.items()}
            token_mask_segment_nsp.append(bert_mlm_nsp)
        return token_mask_segment_nsp
