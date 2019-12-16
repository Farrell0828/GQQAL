import torch 
from torch import nn 
from gquest.encoder import SentenceEncoder 
from gquest.fusioner import Concater 
from gquest.decoder import FCDecoder 


class QuestModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(config['sentence_encoder'])

        if config['fusioner'] == 'concat':
            self.fusioner = Concater()
        else:
            raise NotImplementedError()

        if config['decoder'] == 'fc':
            self.decoder = FCDecoder(config['fc_decoder'])
        else:
            raise NotImplementedError()

    def forward(self, batch):
        question_title_feature = self.sentence_encoder(batch['question_title'])
        question_body_feature = self.sentence_encoder(batch['question_body'])
        answer_feature = self.sentence_encoder(batch['answer'])
        context_features = [
            question_title_feature, question_body_feature, answer_feature
        ]
        fusioned_feature = self.fusioner(context_features)
        output = self.decoder(fusioned_feature)
        return output
