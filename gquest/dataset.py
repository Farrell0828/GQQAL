import torch
import pandas as pd 
import numpy as np 
from sklearn.model_selection import GroupKFold, KFold 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
from transformers import DistilBertTokenizer, BertTokenizer 
from nltk.tokenize import sent_tokenize 


class QuestDataset(Dataset):
    '''
    The Dataset used for Google Quest Challenge.
    '''

    def __init__(self, config, split, fold=-1, overfit=False):
        self.config = config
        self.split = split
        self.fold = fold
        if 'distilbert' in config['tokenizer_type']:
            self.tokenizer = (
                DistilBertTokenizer.from_pretrained(config['tokenizer_type'])
            )
        elif 'bert' in config['tokenizer_type']:
            self.tokenizer = (
                BertTokenizer.from_pretrained(config['tokenizer_type'])
            )
        else:
            raise NotImplementedError()

        if split in ['train', 'val']:
            self.df = pd.read_csv(config['train_csv_path'], index_col=0)
        elif split == 'test':
            self.df = pd.read_csv(config['test_csv_path'], index_col=0)
        else:
            raise NotImplementedError()

        if self.split != 'test':
            self.question_target_cols = [
                'question_asker_intent_understanding',
                'question_body_critical', 'question_conversational',
                'question_expect_short_answer', 'question_fact_seeking',
                'question_has_commonly_accepted_answer',
                'question_interestingness_others', 'question_interestingness_self',
                'question_multi_intent', 'question_not_really_a_question',
                'question_opinion_seeking', 'question_type_choice',
                'question_type_compare', 'question_type_consequence',
                'question_type_definition', 'question_type_entity',
                'question_type_instructions', 'question_type_procedure',
                'question_type_reason_explanation', 'question_type_spelling',
                'question_well_written'
            ]

            self.answer_target_cols = [
                'answer_helpful', 'answer_level_of_information', 
                'answer_plausible', 'answer_relevance',
                'answer_satisfaction', 'answer_type_instructions',
                'answer_type_procedure', 'answer_type_reason_explanation',
                'answer_well_written'
            ]

            self.target_cols = self.question_target_cols + self.answer_target_cols

        if config['split_to_sentences']:
            for col_name in ['question_body', 'answer']:
                self.df[col_name + '_sentences'] = self.df[col_name].apply(
                    lambda x: sent_tokenize(x)[:config['max_n_sentences']]
                )

        if split == 'test' or fold < 0:
            self.qa_ids = self.df.index
        else:
            self.df['question_title'] = self.df['question_title'].astype('category')
            gkf = GroupKFold(n_splits=config['n_folds'])
            train_indexes, val_indexes = list(
                gkf.split(self.df.index, groups=self.df['question_title'])
            )[fold]
            self.qa_ids = (
                self.df.index[train_indexes] 
                if split == 'train' 
                else self.df.index[val_indexes]
            )
        if overfit:
            self.qa_ids = self.qa_ids[:4]

        if config['average_question_targets'] and split == 'train':
            question_mean = self.df.loc[
                self.qa_ids, ['question_title'] + self.question_target_cols
            ].groupby('question_title').mean()
            self.df.loc[self.qa_ids, self.question_target_cols] = (
                self.df.loc[self.qa_ids, ['question_title'] + self.question_target_cols]
                .apply(lambda x: question_mean.loc[x['question_title']], axis=1)
            )

        self.pad_index = self.tokenizer.pad_token_id
        self.feature_cols = ['question_title', 'question_body', 'answer']


    def __len__(self):
        return len(self.qa_ids)
 

    def __getitem__(self, idx):
        qa_id = self.qa_ids[idx]
        item = {}

        token_ids = self.tokenizer.encode(
            self.df.loc[qa_id, 'question_title'], 
            add_special_tokens=True,
            max_length=self.config['max_sequence_length']['question_title'],
            return_tensors='pt'
        )[0]
        item['question_title'] = self._pad_sequence(
            token_ids, 
            self.config['max_sequence_length']['question_title']
        ).long()

        for col_name in ['question_body', 'answer']:
            if self.config['split_to_sentences']:
                token_ids = []
                for sentence in self.df.loc[qa_id, col_name + '_sentences']:
                    one_token_ids = self.tokenizer.encode(
                            sentence,
                            add_special_tokens=True,
                            max_length=self.config['max_sequence_length']['sentence'],
                            return_tensors='pt'
                        )[0]
                    one_token_ids = self._pad_sequence(
                        one_token_ids, 
                        self.config['max_sequence_length']['sentence']
                    ).unsqueeze(0)
                    token_ids.append(one_token_ids)
                item[col_name + '_n_sentences'] = len(token_ids)
                item[col_name] = self._pad_sequence(
                    torch.cat(token_ids, dim=0),
                    self.config['max_n_sentences']
                ).long()
            else:
                token_ids = self.tokenizer.encode(
                    self.df.loc[qa_id, col_name], 
                    add_special_tokens=True,
                    max_length=self.config['max_sequence_length'][col_name],
                    return_tensors='pt'
                )[0]
                item[col_name] = self._pad_sequence(
                    token_ids, 
                    self.config['max_sequence_length'][col_name]
                ).long()

        if self.split != 'test':
            targets = torch.tensor(
                self.df.loc[qa_id, self.target_cols].values.astype('float')
            )
            item['targets'] = targets.float()

        return item


    def _pad_sequence(self, unpadded_sequence, max_sequence_length):
        padded_size = torch.tensor(unpadded_sequence.size()).tolist()
        padded_size[0] = max_sequence_length
        maxpadded_sequences = torch.full(
            size=padded_size,
            fill_value=self.pad_index
        )
        maxpadded_sequences[:unpadded_sequence.size(0)] = unpadded_sequence
        return maxpadded_sequences
