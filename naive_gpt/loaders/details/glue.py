import torch
import random
from torch import nn
from torch.utils import data
from datasets import load_dataset


class GLUEDataset(data.IterableDataset):
    def __init__(self,
                 root: str,
                 subset: str, mode: str,
                 text_transform: callable = None):
        #
        self.dataset = load_dataset(
            'glue', subset, split=mode
        )
        self.text_transform = text_transform

    def __iter__(self):
        return self

    def __next__(self):
        n = len(self.dataset)
        sample = self.dataset[
            random.randrange(n)
        ]
        label = sample['label']

        # sentence
        # question sentence
        # sentence1 sentence2
        if 'question' in sample:
            question = sample['question']
            sentence = sample['sentence']
            if self.text_transform is not None:
                output = self.text_transform(
                    [question, sentence]
                )
        elif 'question2' in sample:
            sentence1 = sample['question1']
            sentence2 = sample['question2']
            if self.text_transform is not None:
                output = self.text_transform(
                    [sentence1, sentence2]
                )
        elif 'sentence2' in sample:
            sentence1 = sample['sentence1']
            sentence2 = sample['sentence2']
            if self.text_transform is not None:
                output = self.text_transform(
                    [sentence1, sentence2]
                )
        else:
            sentence = sample['sentence']
            if self.text_transform is not None:
                output = self.text_transform(sentence)

        #
        label = torch.LongTensor([label])
        input_ids = torch.LongTensor(output['input_ids'])
        token_types = torch.LongTensor(output['token_type_ids'])
        return input_ids, token_types, label
