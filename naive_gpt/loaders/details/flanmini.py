import json
from torch import nn
from torchtext import transforms
from naive_gpt import loaders


class JsonLoader(nn.Module):
    def forward(self, line: str):
        text = json.loads(line)
        return text


class FlanMiniDataset(loaders.LineReader):
    def __init__(self,
                 root: str,
                 mode: str,
                 shuffle: bool = True,
                 min_length: int = 64,
                 buffer_size: int = 16384,
                 text_transform: callable = None):
        #
        datafile = {
            'valid': {
                'flan-mini/flan_mini.jsonl': 1.0
            },
            'train': {
                'flan-mini/flan_mini.jsonl': 1.0
            }
        }
        if mode not in datafile:
            raise RuntimeError
        #
        text_transform = transforms.Sequential(
            JsonLoader(), text_transform
        )
        #
        loaders.LineReader.__init__(
            self, root=root, files=datafile[mode],
            shuffle=shuffle, min_length=min_length,
            buffer_size=buffer_size, return_path=False,
            path_transform=None, text_transform=text_transform
        )
