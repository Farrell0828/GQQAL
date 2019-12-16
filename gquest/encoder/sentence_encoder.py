import torch 
from torch import nn 
from transformers import DistilBertModel, BertModel 


class SentenceEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if 'distilbert' in config['transformer_type']:
            self.transformer = DistilBertModel.from_pretrained(config['transformer_type'])
        elif 'bert' in config['transformer_type']:
            self.transformer = BertModel.from_pretrained(config['transformer_type'])
        else:
            raise NotImplementedError()

        if config['pool_method'] == 'average':
            self.pooler = lambda x: torch.mean(x, dim=1)
        elif config['pool_method'] == 'max':
            self.pooler = lambda x: torch.max(x, dim=1)[0]
        elif config['pool_method'] == 'lstm':
            self.pooler = nn.LSTM(config['transformer_hidden_size'],
                                  config['lstm_hidden_size'],
                                  num_layers=config['lstm_n_layers'],
                                  batch_first=True)
        elif config['pool_method'] not in ['none', 'cls']:
            raise NotImplementedError()

    def forward(self, sentence_indexes):
        '''
        Parameters:
        -----------
        sentence_indexes: torch.Tensor.
            input sequence of size (batch_size, max_sequene_length).

        Retures:
        --------
        sentence_feature: torch.Tensor.
            output sentence feature of size (batch_size, max_sequence_length, 
            feature_size) if pool_method is none, (batch_size, feature_size)
            else.
        '''
        if self.config['pool_method'] == 'cls':
            sentence_feature = self.transformer(sentence_indexes)[1]
        else:
            sentence_feature = self.transformer(sentence_indexes)[0]

        if self.config['pool_method'] in ['average', 'max', 'lstm']:
            sentence_feature = self.pooler(sentence_feature)

        return sentence_feature