import os
import re
import sys
import unicodedata
import pickle

sys.path.append(os.path.abspath('..'))
from config import Config

config = Config()


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2count = {}
        self.word2index = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, '<mask>': 4}
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.n_words = 5
        self.seq_max_len = 0
        self.words = []

    '''处理英文数据的读取方式'''

    def index_englishe_words(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1

    '''处理中文数据读取方式'''

    def index_chinese_words(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1


class DataProcess():
    def __init__(self):
        pass

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.strip())
        s = re.sub(r"([,.!?])", r" \1 ", s)
        s = re.sub(r"s+", r" ", s).strip()
        return s

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def indexes_from_sentence(self, lang, sentence):
        # 前后加上sos和eos。注意句子的句号也要加上，如果这个词没有出现在词典中（已经去除次数小于限定的词），以unk填充
        if type(lang) == dict:
            return [config.sos] + \
                   [lang['word2index'].get(word, config.unk) for word in sentence.split(' ')] + \
                   [config.eos]
        else:
            return [config.sos] + \
                   [lang.word2index.get(word, config.unk) for word in sentence.split(' ')] + \
                   [config.eos]

    def read_data(self, data_path):
        content = open(data_path, encoding='utf-8').read().split('\n')
        # 处理英文数据有用到，能让标点符号和单词分开。中文数据不需要下面这行代码
        # content = [self.normalize_string(s) for s in content]
        return content

    def process(self, data_path):
        content = self.read_data(data_path)
        lang = Lang('chinese')
        for line in content:
            lang.index_chinese_words(line)
        lang.word2count = dict(sorted(lang.word2count.items(), key=lambda x: x[1], reverse=True))
        for key in lang.word2index.keys():
            lang.words.append(key)
        return lang

    def save_vocab(self, vocab_obj):
        with open(config.vocab_path, 'wb') as f:
            pickle.dump(vocab_obj, f)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)

    def save_data_dict(self,data):
        with open('data/data_dict.txt', 'a', encoding='utf-8') as w:
            for key,value in data.items():
                w.write(key+'\t')
                w.write(str(value))
                w.write('\n')
