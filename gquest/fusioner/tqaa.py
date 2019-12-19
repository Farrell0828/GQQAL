import torch 
from torch import nn 
from gquest.encoder import SentenceEncoder 

class TQAAttentionFusioner(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tq_attention = AttentionLayer(
            config['hidden_size'], config['attention_dropout']
        )
        self.qa_attention = AttentionLayer(
            config['hidden_size'], config['attention_dropout']
        )

    def forward(self, features):
        t = features['question_title']
        q = features['question_body']
        a = features['answer']
        q_n_sentences = features['question_body_n_sentences'].long()
        a_n_sentences = features['answer_n_sentences'].long()

        tq_attention_mask = (
            torch.arange(q.size(1))[None, :].to(q_n_sentences.device) 
            < q_n_sentences[:, None]
        )
        qa_attention_mask = (
            torch.arange(a.size(1))[None, :].to(a_n_sentences.device) 
            < a_n_sentences[:, None]
        )

        tq = self.tq_attention(t, q, q, tq_attention_mask)
        tq = tq + t

        qa = self.qa_attention(tq, a, a, qa_attention_mask)
        qa = qa + tq

        return qa


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, attention_dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, attention_mask=None):
        '''
        Parameters:
        -----------
        query: torch.Tensor.
            Size: (batch_size, hidden_size).
        key: torch.Tensor.
            Size: (batch_size, sequence_length, hidden_size).
        value: torch.Tensor.
            Size: (batch_size, sequence_length, hidden_size).
        attention_mask: torch.BoolTensor, optinal.
            Size: (batch_size, sequence_length).

        Retures:
        --------
        outputs: torch.Tensor.
            Size: (batch_size, hidden_size).

        '''
        mixed_query = self.query_linear(query)              # (B, H)
        mixed_key = self.query_linear(key)                  # (B, S, H)
        mixed_value = self.value_linear(value)              # (B, S, H)

        mixed_query = mixed_query.unsqueeze(1)              # (B, 1, H)
        mixed_key = mixed_key.permute(0, 2, 1)              # (B, H, S)
        scores = torch.matmul(mixed_query, mixed_key)       # (B, 1, S)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)    # (B, 1, S)
            scores[~attention_mask] = float('-inf')         # (B, 1, S)
        
        weights = self.softmax(scores)                      # (B, 1, S)
        weights = self.dropout(weights)                     # (B, 1, S)
        outputs = torch.matmul(weights, mixed_value)        # (B, 1, H)
        outputs = outputs.squeeze(1)                        # (B, H)

        return outputs
