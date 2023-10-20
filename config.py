import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = 'data/chinese_data.txt'
    pad_idx = 0  # 填充字符索引位序
    unk_idx = 1  # 未知字符索引位序
    sos_idx = 2  # 开始字符索引位序
    eos_idx = 3  # 结束字符索引位序
    mask_idx = 4
    vocab_path = 'vocab.small'
    IsNext = 1
    NoNext = 0
    no_mask = 0
    # 训练文本的最大长度
    max_predictions_per_seq = 20
    batch_size = 64
    hidden_size = 768
    dropout = 0.1
    layers = 12
    heads = 12
    d_q = 64
    d_k = 64
    d_v = 64

    epochs = 100
    lr = 0.1
    weight_decay = 0.01
    warmup_steps = 8000

