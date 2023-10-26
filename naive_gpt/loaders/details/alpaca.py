import json
from torch import nn
from torchtext import transforms
from naive_torch import loaders


class AlpacaPrompt(nn.Module):
    prompt_1 = 'Below is an instruction that describes a task'
    prompt_2 = ', paired with an input that provides further context'
    prompt_3 = '. Write a response that appropriately completes the request.'

    def forward(self, line: str):
        item = json.loads(line)

        # header
        if item['input'] == '':
            head = self.prompt_1 + self.prompt_3
        else:
            head = self.prompt_1 + self.prompt_2 + self.prompt_3

        # context
        if item['input'] == '':
            text = '{}\n## Instruction\n{}\n## Output\n{}'.format(
                head, item['instruction'], item['output']
            )
        else:
            text = '{}\n## Instruction\n{}\n## Input\n{}\n## Output\n{}'.format(
                head, item['instruction'], item['input'], item['output']
            )
        return text


class AlpacaDataset(loaders.LineReader):
    def __init__(self,
                 root: str,
                 mode: str,
                 shuffle: bool = True,
                 min_length: int = 64,
                 buffer_size: int = 16384,
                 text_transform: callable = None):
        #
        datafile = {
            'test': {
                'alpaca/test.jsonl': 1.0
            },
            'valid': {
                'alpaca/valid.jsonl': 1.0
            },
            'train': {
                'alpaca/train.jsonl': 1.0
            }
        }
        if mode not in datafile:
            raise RuntimeError
        #
        text_transform = transforms.Sequential(
            AlpacaPrompt(), text_transform
            if text_transform is not None
            else nn.Identity()
        )
        #
        loaders.LineReader.__init__(
            self, root=root, files=datafile[mode],
            shuffle=shuffle, min_length=min_length,
            buffer_size=buffer_size, return_path=False,
            path_transform=None, text_transform=text_transform
        )
