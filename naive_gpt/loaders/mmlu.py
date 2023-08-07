import torch
import transformers
from torch.utils import data
import pytorch_lightning as L
from torchtext import transforms
from naive_torch import datasets, layers


class MMLUModule(L.LightningDataModule):
    def __init__(self,
                 root: str,
                 seq_length: int,
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        #
        self.root = root
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        AT = transformers.AutoTokenizer
        self.tokenizer = AT.from_pretrained(
            'facebook/opt-125m'
        )

    def _dataloader(self, mode: str):
        transform = transforms.Sequential(
            layers.FnModule(
                self.tokenizer.encode
            ),
            datasets.Truncate(
                seq_length=self.seq_length,
                output_mode='tail'
            ),
            transforms.ToTensor()
        )
        return data.DataLoader(
            datasets.MMLUDataset(
                self.root, mode=mode,
                text_transform=transform
            ), shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return self._dataloader(mode='valid')
