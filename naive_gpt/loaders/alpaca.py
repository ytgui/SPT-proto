import json
import torch
import transformers
from torch import nn
import lightning as L
from torch.utils import data
from torchtext import transforms
from naive_gpt import layers, loaders


class JsonLoader(nn.Module):
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


class AlpacaDataModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 seq_length: int,
                 batch_size: int,
                 num_workers: int,
                 tokenizer: str):
        super().__init__()
        #
        self.root = root
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        AT = transformers.AutoTokenizer
        if tokenizer == 'opt':
            tokenizer = 'facebook/opt-1.3b'
        elif tokenizer == 'llama':
            tokenizer = 'openlm-research/open_llama_7b'
        self.tokenizer = AT.from_pretrained(tokenizer)
        self.pad_value = self.tokenizer.pad_token_id

    def _dataloader(self, mode: str):
        datafile = {
            'test': {
                'alpaca/test.jsonl': 1.0
            },
            'train': {
                'alpaca/train.jsonl': 1.0
            },
            'valid': {
                'alpaca/valid.jsonl': 1.0
            }
        }
        #
        transform = transforms.Sequential(
            JsonLoader(),
            layers.FnModule(
                self.tokenizer.encode
            ),
            loaders.ClampPadding(
                seq_length=self.seq_length,
                pad_value=self.pad_value
            ),
            transforms.ToTensor()
        )
        return data.DataLoader(
            loaders.LineReader(
                root=self.root,
                files=datafile[mode],
                shuffle=True,
                text_transform=transform
            ), shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return self._dataloader(mode='valid')

    def train_dataloader(self):
        return self._dataloader(mode='train')

    def predict_dataloader(self):
        return self._dataloader(mode='test')
