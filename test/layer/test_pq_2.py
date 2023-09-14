import torch
from torch import nn, optim
from torch.utils import data
import pytorch_lightning as L
from torch.optim import lr_scheduler as lr
from pytorch_lightning import callbacks as cb
from torchmetrics import Accuracy
from naive_gpt import layers
from sklearn import datasets


class DataModule(L.LightningDataModule):
    N_VALID = 50_000
    N_TRAIN = 50_000
    N_SAMPLE = N_TRAIN + N_VALID

    def __init__(self,
                 n_centers: int,
                 n_features: int,
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        #
        self.batch_size = batch_size
        self.num_workers = num_workers
        #
        blobs = datasets.make_blobs(
            self.N_SAMPLE, n_features=n_features,
            centers=n_centers, center_box=(-1.0, 1.0),
            cluster_std=0.1, return_centers=True
        )
        assert len(blobs) == 3
        x = torch.FloatTensor(blobs[0])
        label = torch.IntTensor(blobs[1])
        centers = torch.FloatTensor(blobs[2])
        y = torch.index_select(
            centers, dim=0, index=label
        )
        #
        samples = list(zip(x, y))
        self.test_data = samples[-self.N_VALID:]
        self.train_data = samples[:self.N_TRAIN]

    def _dataloader(self, mode: str):
        if mode == 'valid':
            dataset = self.test_data
        elif mode == 'train':
            dataset = self.train_data
        else:
            raise NotImplementedError
        #
        return data.DataLoader(
            data.ConcatDataset([
                dataset
                for _ in range(self.batch_size)
            ]), shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def train_dataloader(self):
        return self._dataloader(mode='train')

    def val_dataloader(self):
        return self._dataloader(mode='valid')


class LightningModel(L.LightningModule):
    def __init__(self,
                 d_model: int,
                 n_tables: int,
                 n_classes: int,
                 method: str):
        super().__init__()
        #
        self.lr = 1e-2
        self.weight_decay = 1e-4
        #
        assert d_model % n_tables == 0
        d_codeword = d_model // n_tables
        self.quantizer = layers.PQ(
            d_codeword=d_codeword, n_codewords=n_classes,
            n_subspaces=n_tables, method=method
        )
        self.accuracy_fn = Accuracy(
            task='multiclass', num_classes=n_classes
        )
        self.distance_fn = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = lr.ExponentialLR(
            optimizer, gamma=0.5
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: tuple, batch_idx: int):
        x, center = batch
        loss_pq = self.quantizer('train', x)
        self.log('loss_pq', loss_pq, prog_bar=True)
        return loss_pq

    def validation_step(self, batch: tuple, batch_idx: int):
        x, center = batch
        accuracy = self.accuracy_fn(
            self.quantizer('encode', x),
            target=self.quantizer('encode', center)
        )
        center_error = self.distance_fn(
            self.quantizer('quantize', x), target=center
        )
        self.log('accuracy@1', accuracy, prog_bar=True)
        self.log('center_error', center_error, prog_bar=True)


def test_blobs_pq():
    n_centers = 64
    n_features = 16
    batch_size = 256
    n_tables = 4

    # loader
    dm = DataModule(
        n_centers=n_centers, n_features=n_features,
        batch_size=batch_size, num_workers=1
    )

    # lightning
    model = LightningModel(
        d_model=n_features, n_classes=n_centers,
        n_tables=n_tables, method='k-means'
    )
    summary = cb.ModelSummary(3)
    trainer = L.Trainer(
        precision='32-true', accelerator='gpu', devices=1,
        max_epochs=2, limit_val_batches=64, limit_train_batches=1024,
        gradient_clip_val=1.0, accumulate_grad_batches=1, callbacks=[summary]
    )

    # training
    trainer.fit(model, dm)
    evaluation = trainer.validate(
        model, dm, verbose=True
    )[0]
    assert evaluation['accuracy@1'] >= 0.75
    assert evaluation['center_error'] <= 0.20

    #
    print('[PASS] test_blobs_pq()')


def main():
    test_blobs_pq()


if __name__ == "__main__":
    main()
