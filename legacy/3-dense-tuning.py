import os
import torch
import argparse
import lightning as L
from deepspeed import ops
from torch import nn, optim
from torch.optim import lr_scheduler as lr
from lightning.pytorch import callbacks, strategies
from naive_gpt import loaders, models, utils
from torchmetrics import Perplexity


class LightningModel(L.LightningModule):
    def __init__(self,
                 d_lora: int,
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
        if 'opt' in ckpt_path:
            self.model = models.OPTModel(**config)
        elif 'llama' in ckpt_path:
            self.model = models.LLaMAModel(**config)
        else:
            raise RuntimeError
        self.model.load_state_dict(ckpt['state_dict'])
        # insert LoRA
        if d_lora > 0:
            upgrader = utils.ModuleUpgrader(
                handler=utils.LoRAHandler(
                    d_lora=d_lora
                )
            )
            self.model = upgrader.visit(self.model)
        # loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.ppl_fn = Perplexity(
            # ignore_index=PAD_VALUE
        )

    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        optimizer = ops.adam.DeepSpeedCPUAdam(
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
        output: torch.Tensor = self.model(src)
        loss = self.loss_fn(
            output.flatten(end_dim=-2),
            target=target.flatten()
        )
        return output, loss

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
            batch[:, 1:-1], target=batch[:, 2:]
        )[0]
        # ppl
        self.ppl_fn.to(batch.device)
        self.log(
            'ppl', self.ppl_fn(
                output, target=batch[:, 2:]
            ),
            prog_bar=True, sync_dist=True
        )
        # mmlu
        position = batch[:, 0]
        target = target = batch[:, position]
        position_m2 = torch.subtract(position, 2)
        predict = torch.argmax(
            output[:, position_m2, :], dim=-1
        )
        accuracy = torch.mean(
            torch.eq(predict, target).type(torch.float)
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
        default='.data/opt-1.3b.ckpt'
    )
    parser.add_argument(
        '--seq_length', help='pad sequence to fixed length',
        default=256
    )
    parser.add_argument(
        '--batch_size', help='specify batch size',
        default=1
    )
    parser.add_argument(
        '--n_devices', help='number of gpus to use',
        default=4
    )
    parser.add_argument(
        '--d_lora', help='dim oflow rank adaptation',
        default=16
    )
    args = parser.parse_args()

    # loader
    if str(args.ckpt).find('opt') != -1:
        tokenizer = 'opt'
    elif str(args.ckpt).find('llama') != -1:
        tokenizer = 'llama'
    else:
        raise NotImplementedError
    dm = loaders.MMLUDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        n_shots=5, batch_size=args.batch_size,
        num_workers=1, tokenizer=tokenizer,
        seq_length=args.seq_length + 1
    )

    # lightning
    model = LightningModel(
        d_lora=args.d_lora,
        ckpt_path=args.ckpt
    )
    summary = callbacks.ModelSummary(3)
    trainer = L.Trainer(
        strategy=strategies.DeepSpeedStrategy(
            stage=3, offload_optimizer=True,
            offload_parameters=True, cpu_checkpointing=True
        ),
        precision='32-true', accelerator='cuda', devices=args.n_devices,
        max_epochs=20, limit_train_batches=256, limit_val_batches=64,
        accumulate_grad_batches=1, gradient_clip_val=1.0, callbacks=[summary]
    )

    # fine-tuning
    trainer.fit(model, dm)
    trainer.validate(model, dm)


if __name__ == '__main__':
    main()
