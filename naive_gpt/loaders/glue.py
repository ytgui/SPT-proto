import torch
import transformers
import lightning as L
from torch.utils import data
from torchtext import transforms
from naive_gpt import loaders, layers
from .details.glue import GLUEDataset


class GLUEDataModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 subset: int,
                 seq_length: int,
                 batch_size: int,
                 num_workers: int,
                 tokenizer: str):
        super().__init__()
        #
        self.root = root
        self.subset = subset
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        AT = transformers.AutoTokenizer
        self.tokenizer = AT.from_pretrained(tokenizer)
        self.pad_value = self.tokenizer.pad_token_id or 0

    def _dataloader(self, mode: str):
        transform = transforms.Sequential(
            layers.FnModule(
                self.tokenizer.encode_plus,
                return_token_type_ids=True,
                max_length=self.seq_length,
                pad_to_max_length=True
            )
        )
        return data.DataLoader(
            GLUEDataset(
                self.root,
                subset=self.subset,
                mode=mode,
                text_transform=transform
            ),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self._dataloader(mode='validation')

    def train_dataloader(self):
        return self._dataloader(mode='train')

    def test_dataloader(self):
        return self._dataloader(mode='test')
