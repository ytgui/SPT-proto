import re
import torch
import random
from torch import nn


class Sanitize(nn.Module):
    def _remove_blanks(self, x: str):
        blanks = [
            # extra spaces
            [r'\s+', r' '],
            [r'^\s+', r''],
            [r'\s+$', r''],
        ]
        for pattern, repl in blanks:
            x = re.sub(pattern, repl, x)
        return x

    def forward(self, text: str):
        assert isinstance(text, str)
        #
        normalizes = [
            # empty quote
            [r'\(\)', r' '],
            [r'\[\]', r' '],
            [r'\{\}', r' '],
            # incorrect punctuation
            [r'\s([\,\.\?\!\;\:])', r'\g<1>']
        ]
        clean_text = []
        for paragraph in text.split('\n\n'):
            for pattern, repl in normalizes:
                paragraph = re.sub(pattern, repl, paragraph)
                paragraph = self._remove_blanks(paragraph)
            if len(paragraph) > 0:
                clean_text.append(paragraph)
        return '\n\n'.join(clean_text)


class ClampPadding(nn.Module):
    def __init__(self,
                 seq_length: int,
                 pad_value: int = 0):
        nn.Module.__init__(self)
        #
        self.pad_value = pad_value
        self.seq_length = seq_length

    def forward(self, sequence: list):
        assert isinstance(
            sequence, (list, tuple)
        )
        #
        n_sequence = len(sequence)
        if n_sequence < self.seq_length:
            sequence.extend([
                self.pad_value for _ in
                range(self.seq_length - n_sequence)
            ])
        elif n_sequence > self.seq_length:
            left = random.randrange(
                n_sequence - self.seq_length + 1
            )
            right = left + self.seq_length
            sequence = sequence[left:right]
        return sequence


class TruncPadding(nn.Module):
    def __init__(self,
                 seq_length: int,
                 pad_value: int = 0):
        nn.Module.__init__(self)
        #
        self.pad_value = pad_value
        self.seq_length = seq_length

    def forward(self, sequence: list):
        assert isinstance(
            sequence, (list, tuple)
        )
        #
        n_sequence = len(sequence)
        if n_sequence < self.seq_length:
            sequence.extend([
                self.pad_value for _ in
                range(self.seq_length - n_sequence)
            ])
        elif n_sequence > self.seq_length:
            sequence = sequence[-self.seq_length:]
            n_sequence = len(sequence)
        return [n_sequence] + sequence
