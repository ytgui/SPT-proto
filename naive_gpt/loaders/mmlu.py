import torch
import transformers
import lightning as L
from torch.utils import data
from torchtext import transforms
from naive_gpt import loaders, layers
from .details.mmlu import MMLUDataset


class MMLUDataModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 n_shots: int,
                 max_length: int,
                 batch_size: int,
                 num_workers: int,
                 tokenizer='opt'):
        super().__init__()
        #
        self.root = root
        self.n_shots = n_shots
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        AT = transformers.AutoTokenizer
        if tokenizer == 'opt':
            tokenizer = 'facebook/opt-1.3b'
        elif tokenizer == 'llama':
            tokenizer = 'princeton-nlp/Sheared-LLaMA-2.7B'
        self.tokenizer = AT.from_pretrained(tokenizer)
        self.pad_value = self.tokenizer.pad_token_id

    def _dataloader(self, mode: str):
        transform = transforms.Sequential(
            layers.FnModule(
                self.tokenizer.encode
            ),
            loaders.TruncPadding(
                seq_length=self.max_length,
                pad_value=self.pad_value
            ),
            transforms.ToTensor()
        )
        return data.DataLoader(
            MMLUDataset(
                self.root, mode=mode,
                n_shots=self.n_shots,
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

