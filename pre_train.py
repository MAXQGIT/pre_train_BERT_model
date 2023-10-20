from data.data_process import DataProcess
from config import Config
from bert.bert_data import BertData
from torch.utils.data import DataLoader
from bert.pre_training_bert import PreTrainingBert
from bert.bert import Bert
from bert.utils import SpecialOptimizer
from torch import nn, optim
config = Config()


def main():
    data_obj = DataProcess()
    data = data_obj.process(config.data_path)
    data_obj.save_data_dict(data.word2index)
    data_obj.save_vocab(data)
    train_data_obj = BertData(config.data_path, data)
    train_data = train_data_obj.build_mlm_nsp()
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    bert = Bert(data.n_words,config.hidden_size)
    bert = bert.to(config.device)
    pre_training = PreTrainingBert(bert,data.n_words,config.hidden_size)
    special_optim = SpecialOptimizer(  # 变化学习率的优化器
        optimizer=optim.Adam(
            params=pre_training.parameters(), lr=config.lr, weight_decay=config.weight_decay),
        warmup_steps=config.warmup_steps,
        d_model=config.hidden_size)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    pre_training.pre_training(special_optim, criterion, train_data_loader)



if __name__ == '__main__':
    main()
