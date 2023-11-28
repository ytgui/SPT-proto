import torch
import transformers
import lightning as L
from torch.utils import data
from torchtext import transforms
from naive_gpt import loaders, layers
from .details.flanmini import FlanMiniDataset
from .details.concat import ConcatDataset
from .details.mmlu import MMLUDataset


class MMLUDataModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 n_shots: int,
                 seq_length: int,
                 batch_size: int,
                 num_workers: int,
                 tokenizer='opt'):
        super().__init__()
        #
        self.root = root
        self.n_shots = n_shots
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        AT = transformers.AutoTokenizer
        if tokenizer == 'opt':
            tokenizer = 'facebook/opt-1.3b'
        elif tokenizer == 'llama':
            tokenizer = 'princeton-nlp/Sheared-LLaMA-2.7B'
        self.tokenizer = AT.from_pretrained(tokenizer)
        self.pad_value = self.tokenizer.pad_token_id or 0

    def _dataloader(self, mode: str):
        transform = transforms.Sequential(
            layers.FnModule(
                self.tokenizer.encode
            ),
            loaders.TruncPadding(
                seq_length=self.seq_length,
                pad_value=self.pad_value
            ),
            transforms.ToTensor()
        )
        if mode == 'train':
            dataset = ConcatDataset({
                MMLUDataset(
                    self.root, mode=mode,
                    n_shots=self.n_shots,
                    text_transform=transform
                ): 0.1,
                FlanMiniDataset(
                    self.root, mode=mode,
                    text_transform=transform
                ): 1.0
            })
        else:
            dataset = MMLUDataset(
                self.root, mode=mode,
                n_shots=self.n_shots,
                text_transform=transform
            )
        return data.DataLoader(
            dataset, shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return self._dataloader(mode='valid')

    def train_dataloader(self):
        return self._dataloader(mode='train')

    def test_dataloader(self):
        return self._dataloader(mode='test')
