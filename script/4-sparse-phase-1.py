import os
import torch
import argparse
import lightning as L
from torch import nn, optim
from torch.optim import lr_scheduler as lr
from lightning.pytorch import callbacks, strategies
from naive_gpt import loaders, models, utils
from torchmetrics import Perplexity
from torchmetrics import Accuracy


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
        upgrader = utils.ModuleUpgrader(
            handler=utils.SparseLoRAHandler(
                lora_r=d_lora,
                lora_dropout=p_dropout,
                stage=1
            )
        )
        self.model = upgrader.visit(model)
        # loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.ppl_fn = Perplexity(
            ignore_index=self.PAD_VALUE
        )

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = lr.ExponentialLR(
            optimizer, gamma=0.9
        )
        return [optimizer], [scheduler]

    def shared_step(self,
                    src: torch.Tensor,
                    target: torch.Tensor):
        output = self.model(src)
        loss = self.loss_fn(
            output.flatten(end_dim=-2),
            target=target.flatten()
        )
        #
        loss_pq = 0.0
        for name, buffer in self.named_buffers():
            if not name.endswith('.loss'):
                continue
            loss_pq += buffer
        #
        return output, loss + 0.1 * loss_pq

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
        # ppl
        self.ppl_fn.to(batch.device)
        self.log(
            'ppl', self.ppl_fn(
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
        '--seq_length', help='pad sequence to fixed length',
        default=256
    )
    parser.add_argument(
        '--batch_size', help='specify batch size',
        default=8
    )
    parser.add_argument(
        '--n_devices', help='number of gpus to use',
        default=2
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
    if str(args.ckpt).find('opt') != -1:
        tokenizer = 'opt'
    elif str(args.ckpt).find('llama') != -1:
        tokenizer = 'llama'
    else:
        raise NotImplementedError
    dm = loaders.WikitextDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=args.seq_length + 1,
        batch_size=args.batch_size,
        num_workers=1, tokenizer=tokenizer
    )

    # lightning
    model = LightningModel(
        d_lora=args.d_lora,
        p_dropout=args.p_dropout,
        ckpt_path=args.ckpt
    )
    summary = callbacks.ModelSummary(3)
    trainer = L.Trainer(
        strategy=strategies.FSDPStrategy(
            use_orig_params=True, cpu_offload=True
        ),
        precision='32-true', accelerator='cuda', devices=args.n_devices,
        max_epochs=20, limit_train_batches=1024, limit_val_batches=64,
        callbacks=[summary]
    )

    # fine-tuning
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
