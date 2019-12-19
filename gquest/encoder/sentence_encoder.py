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
            self.pooler = lambda x: torch.mean(x, dim=-2)
        elif config['pool_method'] == 'max':
            self.pooler = lambda x: torch.max(x, dim=-2)[0]
        elif config['pool_method'] == 'lstm':
            self.pooler = nn.LSTM(config['transformer_hidden_size'],
                                  config['lstm_hidden_size'],
                                  num_layers=config['lstm_n_layers'],
                                  batch_first=True)
        elif config['pool_method'] not in ['none', 'cls']:
            raise NotImplementedError()

    def forward(self, sentence_indexes, ndim=2):
        '''
        Parameters:
        -----------
        sentence_indexes: torch.Tensor.
            input sequences of size (batch_size, max_sequene_length) if ndim=2 
            or (batch_size, max_n_sequences, max_sequence_length) if ndim=3.

        Retures:
        --------
        sentence_feature: torch.Tensor.
            output sentence feature of size (batch_size, max_sequence_length, 
            feature_size) if pool_method is none, (batch_size, feature_size)
            else when input sequences's ndim is 3, (batch_size, max_n_sequences 
            max_sequene_length, feature_size) if pool_method is none, 
            (batch_size, max_n_sequences, feature_size) else. 
        '''
        if ndim == 2:
            sentence_feature = self.transformer(sentence_indexes)[
                1 if self.config['pool_method'] == 'cls' else 0
            ]
        elif ndim == 3:
            batch_size, n_seq, seq_len = sentence_indexes.size()
            sentence_feature = sentence_indexes.view(-1, seq_len)
            if self.config['pool_method'] == 'cls':
                sentence_feature = self.transformer(sentence_feature)[1]
                sentence_feature = sentence_feature.view(batch_size, n_seq, -1)
            else:
                sentence_feature = self.transformer(sentence_feature)[0]
                sentence_feature = sentence_feature.view(
                    batch_size, n_seq, seq_len, -1
                )
        else:
            raise ValueError('Number of dim of input sequence must be 2 or 3.')

        if self.config['pool_method'] in ['average', 'max', 'lstm']:
            sentence_feature = self.pooler(sentence_feature)

        return sentence_feature