import torch 
from torch import nn 
from gquest.encoder import SentenceEncoder 
from gquest.fusioner import Concater, TQAAttentionFusioner 
from gquest.decoder import FCDecoder 


class QuestModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(config['sentence_encoder'])

        if config['fusioner'] == 'concat':
            self.fusioner = Concater()
        elif config['fusioner'] == 'tqaa':
            self.fusioner = TQAAttentionFusioner(config['tqaa_fusioner'])
        else:
            raise NotImplementedError()

        if config['decoder'] == 'fc':
            self.decoder = FCDecoder(config['fc_decoder'])
        else:
            raise NotImplementedError()

        self.ndim = config['input_ndim']

    def forward(self, batch):
        qt_feature = self.sentence_encoder(batch['question_title'], ndim=2)
        qb_feature = self.sentence_encoder(batch['question_body'], ndim=self.ndim)
        a_feature = self.sentence_encoder(batch['answer'], ndim=self.ndim)
        context_features = {
            'question_title': qt_feature,
            'question_body': qb_feature,
            'answer': a_feature
        }
        if self.ndim == 3:
            context_features['question_body_n_sentences'] = (
                batch['question_body_n_sentences']
            )
            context_features['answer_n_sentences'] = (
                batch['answer_n_sentences']
            )
        fusioned_feature = self.fusioner(context_features)
        output = self.decoder(fusioned_feature)
        return output
