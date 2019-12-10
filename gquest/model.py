import torch 
from torch import nn 
from transformers import DistilBertModel, BertModel 

class QuestModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'distilbert' in config['transformer_type']:
            self.transformer = DistilBertModel.from_pretrained(config['transformer_type'])
        elif 'bert' in config['transformer_type']:
            self.transformer = BertModel.from_pretrained(config['transformer_type'])
        else:
            raise NotImplementedError()
        self.fc = nn.Linear(
            3*config['transformer_hidden_size'], config['output_size']
        )
        if config['pool_method'] == 'average':
            self.pooler = torch.mean
        elif config['pool_method'] == 'max':
            self.pooler = torch.max
        else:
            raise NotImplementedError()
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        # (batch_size, max_question_title_length, transformer_hidden_size)
        question_title_feature = self.transformer(batch['question_title'])[0]
        # (batch_size, transformer_hidden_size)
        question_title_feature = self.pooler(question_title_feature, 1)
        # (batch_size, max_question_body_length, transformer_hidden_size)
        question_body_feature = self.transformer(batch['question_body'])[0]
        # (batch_size, transformer_hidden_size)
        question_body_feature = self.pooler(question_body_feature, 1)
        # (batch_size, max_answer_length, transformer_hidden_size)
        answer_feature = self.transformer(batch['answer'])[0]
        # (batch_size, transformer_hidden_size)
        answer_feature = self.pooler(answer_feature, 1)
        # (batch_size, 3*transformer_hidden_size)
        context_features = torch.cat([
            question_title_feature, 
            question_body_feature, 
            answer_feature
        ], -1)
        # (batch_size, output_size)
        output = self.fc(context_features)
        # (batch_size, output_size)
        output = self.sigmoid(output)
        return output
