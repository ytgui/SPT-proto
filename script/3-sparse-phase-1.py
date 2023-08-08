import os
import torch
import argparse
from torch import nn, optim
import pytorch_lightning as L
from torch.optim import lr_scheduler as lr
from naive_torch.models import ModuleUpgrader
from naive_gpt import loaders, models, tuning
from pytorch_lightning import callbacks
from torchmetrics import Perplexity


class LightningModel(L.LightningModule):
    PAD_VALUE = 0x01

    def __init__(self,
                 d_lora: int,
                 p_dropout: int,
                 ckpt_path: str):
        super().__init__()
        # optim
        self.lr = 1e-4
        self.weight_decay = 1e-2
        # checkpoint
        ckpt = torch.load(
            f=ckpt_path
        )
        config = ckpt['config']
        model = models.OPTModel(**config)
        model.load_state_dict(ckpt['state_dict'])
        # insert LoRA
        upgrader_1 = ModuleUpgrader(
            handler=tuning.LoRAUpgrader(
                lora_r=d_lora,
                lora_dropout=p_dropout
            )
        )
        model = upgrader_1.visit(model)
        # insert quantizer
        upgrader_2 = ModuleUpgrader(
            handler=tuning.QuantizedUpgrader(
                d_codeword=4, n_codewords=64
            )
        )
        self.model = upgrader_2.visit(model)
        # loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fn = Perplexity(
            ignore_index=self.PAD_VALUE
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = lr.ExponentialLR(
            optimizer, gamma=0.5
        )
        return [optimizer], [scheduler]

    def shared_step(self,
                    src: torch.Tensor,
                    target: torch.Tensor):
        output = self.model(src)
        loss = self.loss_fn(
            torch.flatten(
                output, end_dim=-2
            ),
            target=torch.flatten(target)
        )
        #
        loss_vq = 0.0
        for name, buffer in self.named_buffers():
            if not name.endswith('.loss'):
                continue
            loss_vq += buffer
        #
        return output, loss + 0.1 * loss_vq

    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int):
        assert batch.dim() == 2
        loss = self.shared_step(
            batch[:, :-1], target=batch[:, 1:]
        )[-1]
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int):
        assert batch.dim() == 2
        output = self.shared_step(
            batch[:, :-1], target=batch[:, 1:]
        )[0]
        self.log(
            'ppl', self.metrics_fn(
                output, target=batch[:, 1:]
            ),
            prog_bar=True, sync_dist=True
        )


def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', help='specify model path',
        default='.data/opt-125m.ckpt'
    )
    parser.add_argument(
        '--device', help='device of cpu or cuda',
        default='cpu'
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
        '--d_lora', help='dim oflow rank adaptation',
        default=16
    )
    parser.add_argument(
        '--p_dropout', help='dropout probability of lora',
        default=0.1
    )
    args = parser.parse_args()
    print('[INFO] args:', vars(args))

    # loader
    dm = loaders.AlpacaDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=args.seq_length + 1,
        batch_size=args.batch_size,
        num_workers=1
    )

    # lightning
    model = LightningModel(
        d_lora=args.d_lora,
        p_dropout=args.p_dropout,
        ckpt_path=args.ckpt
    )
    summary = callbacks.ModelSummary(3)
    trainer = L.Trainer(
        precision='32-true', accelerator=args.device, devices=1,
        max_epochs=20, limit_train_batches=1024, limit_val_batches=64,
        callbacks=[summary]
    )

    # fine-tuning
    trainer.validate(model, dm)


if __name__ == '__main__':
    main()
