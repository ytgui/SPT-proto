import os
import torch
import argparse
import lightning as L
from torch import nn, optim
from torch.optim import lr_scheduler as lr
from lightning.pytorch import callbacks
from naive_gpt import loaders, models, utils
from torchmetrics import Accuracy


class LightningModel(L.LightningModule):
    def __init__(self,
                 d_lora: int,
                 n_classes: int,
                 ckpt_path: str):
        super().__init__()
        # optim
        self.lr = 1e-4
        self.weight_decay = 1e-1
        # checkpoint
        ckpt = torch.load(
            f=ckpt_path
        )
        config = ckpt['config']
        if 'bert' in ckpt_path:
            self.model = models.BertModel(**config)
        else:
            raise RuntimeError
        self.model.load_state_dict(ckpt['state_dict'])
        self.cls = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(
                config['d_model'], n_classes
            )
        )
        # model adapter
        for stage in ['lora', 'pq-v2']:
            upgrader = utils.ModuleUpgrader(
                handler=utils.SparseLoRAHandler(
                    d_lora=d_lora, stage=stage
                )
            )
            self.model = upgrader.visit(self.model)
        # loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy(
            'multiclass', num_classes=n_classes
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = lr.ExponentialLR(
            optimizer, gamma=0.9
        )
        return [optimizer], [scheduler]

    def shared_step(self,
                    tokens: torch.Tensor,
                    types: torch.Tensor,
                    labels: torch.Tensor):
        logit = self.model(
            tokens, token_types=types
        )
        output = self.cls(
            torch.mean(logit, dim=1)
        )
        loss = self.loss_fn(
            output, target=labels.flatten()
        )
        return output, loss

    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int):
        assert len(batch) == 3
        tokens, types, labels = batch
        #
        if batch_idx % 1 == 0:
            true_tensor = torch.scalar_tensor(
                True, dtype=torch.bool
            )
            for name, trigger in self.named_buffers():
                if not name.endswith('.trigger'):
                    continue
                trigger.fill_(true_tensor)
        #
        loss = self.shared_step(
            tokens, types, labels
        )[-1]
        #
        loss_aux = 0.0
        if batch_idx % 1 == 0:
            for name, buffer in self.named_buffers():
                if not name.endswith('.loss'):
                    continue
                loss_aux += buffer
        #
        loss += 1e-2 * loss_aux
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self,
                        batch: list,
                        batch_idx: int):
        assert len(batch) == 3
        tokens, types, labels = batch
        output, loss = self.shared_step(
            tokens, types, labels
        )
        # loss
        self.log(
            'loss', loss, sync_dist=True
        )
        # glue
        accuracy = self.accuracy_fn(
            output, target=labels.flatten()
        )
        self.log(
            'accuracy', accuracy,
            prog_bar=True, sync_dist=True
        )


def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', help='specify model path',
        default='.data/bert-base-uncased.ckpt'
    )
    parser.add_argument(
        '--seq_length', help='pad sequence to fixed length',
        default=256
    )
    parser.add_argument(
        '--batch_size', help='specify batch size',
        default=16
    )
    parser.add_argument(
        '--n_accumulate', help='specify accumulate size',
        default=1
    )
    parser.add_argument(
        '--n_devices', help='number of gpus to use',
        default=1
    )
    parser.add_argument(
        '--d_lora', help='dim oflow rank adaptation',
        default=16
    )
    args = parser.parse_args()
    print('[INFO] args:', vars(args))

    # loader
    if str(args.ckpt).find('bert') != -1:
        tokenizer = 'bert-base-uncased'
    else:
        raise NotImplementedError
    dm = loaders.GLUEDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        num_workers=1,
        subset='cola'
    )

    # lightning
    model = LightningModel(
        d_lora=args.d_lora,
        ckpt_path=args.ckpt,
        n_classes=2
    )
    summary = callbacks.ModelSummary(3)
    checker = callbacks.ModelCheckpoint(
        save_last=True, save_top_k=3,
        monitor='accuracy', mode='max',
        filename='LM-{epoch}-{accuracy:.3f}'
    )
    trainer = L.Trainer(
        precision='32-true', accelerator='cuda', devices=args.n_devices,
        max_epochs=20, limit_train_batches=args.n_accumulate * 256, limit_val_batches=args.n_accumulate * 64,
        accumulate_grad_batches=args.n_accumulate, gradient_clip_val=1.0, callbacks=[summary, checker]
    )

    # fine-tuning
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
