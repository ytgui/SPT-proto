import torch
import transformers
from torch.utils import data
import pytorch_lightning as L
from torchtext import transforms
from naive_gpt import loaders, layers


class WikitextDataModule(L.LightningDataModule):
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

    def _dataloader(self, mode: str):
        datafile = {
            'test': {
                'wikitext-103/wiki.test.raw': 1.0
            },
            'train': {
                'wikitext-103/wiki.train.raw': 1.0
            },
            'valid': {
                'wikitext-103/wiki.valid.raw': 1.0
            }
        }
        #
        transform = transforms.Sequential(
            layers.FnModule(
                self.tokenizer.encode
            ),
            loaders.ClampPadding(
                seq_length=self.seq_length,
                pad_value=0x01
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
