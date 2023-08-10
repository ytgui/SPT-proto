import os
import torch
from torch import nn, optim
import pytorch_lightning as L
from torch.optim import lr_scheduler as lr
from pytorch_lightning import callbacks
from naive_gpt import loaders, models
from torchmetrics import Accuracy


class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        #
        self.lr = 1e-4
        self.weight_decay = 1e-2
        #
        ckpt = torch.load(
            '.data/opt-125m.ckpt'
        )
        config = ckpt['config']
        self.model = models.OPTModel(**config)
        self.model.load_state_dict(ckpt['state_dict'])
        #
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fn = Accuracy(task='binary')

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = lr.ExponentialLR(
            optimizer, gamma=0.5
        )
        return [optimizer], [scheduler]

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int):
        src = batch[:, :-1]
        target = batch[:, -1]
        output = self.model(src)[:, -1]
        predict = torch.argmax(output, dim=-1)
        is_equal = torch.eq(predict, target)
        accuracy = self.metrics_fn(
            is_equal, target=torch.ones_like(is_equal)
        )
        self.log('length', src.size(1), prog_bar=True)
        self.log('accuracy', accuracy, prog_bar=True)


def train():
    seq_length = 256
    batch_size = 1
    n_shots = 1

    # loader
    dm = loaders.MMLUDataModule(
        root=os.getenv('HOME') + '/Public/Datasets/text/',
        n_shots=n_shots, max_length=seq_length + 1,
        batch_size=batch_size, num_workers=1
    )

    # lightning
    model = LightningModel()
    summary = callbacks.ModelSummary(3)
    trainer = L.Trainer(
        precision='32-true', accelerator='cpu', devices=1,
        limit_val_batches=256, callbacks=[summary]
    )

    # evaluate
    evaluation = trainer.validate(model, dm)[0]
    assert evaluation['length'] <= seq_length
    assert evaluation['accuracy'] >= 0.1


def main():
    train()


if __name__ == "__main__":
    main()