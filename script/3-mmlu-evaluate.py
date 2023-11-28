import os
import torch
import argparse
import lightning as L
from lightning.pytorch import callbacks
from naive_gpt import loaders, models, utils
from torchmetrics import Perplexity


class LightningModel(L.LightningModule):
    def __init__(self,
                 d_lora: int,
                 ckpt_path: str,
                 spt_ckpt_path: str):
        super().__init__()
        #
        self.model = self.init_model(
            d_lora=d_lora,
            ckpt_path=ckpt_path,
            spt_ckpt_path=spt_ckpt_path
        )
        self.ppl_fn = Perplexity(
            # ignore_index=PAD_VALUE
        )

    def init_model(self,
                   d_lora: int,
                   ckpt_path: str,
                   spt_ckpt_path: str):
        # load
        ckpt = torch.load(
            f=ckpt_path
        )
        config = ckpt['config']
        if 'opt' in ckpt_path:
            model = models.OPTModel(**config)
        elif 'llama' in ckpt_path:
            model = models.LLaMAModel(**config)
        else:
            raise RuntimeError
        model.load_state_dict(ckpt['state_dict'])

        # model adapter
        for stage in ['lora', 'ffn', 'mha_v1', 'mha_v2']:
            upgrader = utils.ModuleUpgrader(
                handler=utils.SparseLoRAHandler(
                    d_lora=d_lora, stage=stage
                )
            )
            model = upgrader.visit(model)

        # load lora
        spt_ckpt = torch.load(
            f=spt_ckpt_path
        )
        output = model.load_state_dict(
            spt_ckpt['state_dict'], strict=False
        )
        for name in output.missing_keys:
            assert 'lora' not in name

        #
        return model

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        assert batch.dim() == 2
        output = self.model(batch[:, 1:-1])
        # ppl
        self.ppl_fn.to(batch.device)
        self.log(
            'ppl', self.ppl_fn(
                output.type(torch.float),
                target=batch[:, 2:]
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
        default='.data/opt-125m.ckpt'
    )
    parser.add_argument(
        '--spt_ckpt', help='specify lora path',
        default='.data/opt-125m-spt.ckpt'
    )
    parser.add_argument(
        '--seq_length', help='pad sequence to fixed length',
        default=512
    )
    parser.add_argument(
        '--batch_size', help='specify batch size',
        default=1
    )
    parser.add_argument(
        '--test_batches', help='specify test batches',
        default=64
    )
    parser.add_argument(
        '--d_lora', help='dim oflow rank adaptation',
        default=16
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
        ckpt_path=args.ckpt,
        spt_ckpt_path=args.spt_ckpt
    )
    summary = callbacks.ModelSummary(3)

    # predict
    trainer = L.Trainer(
        precision='32-true', accelerator='cuda', devices=1,
        limit_test_batches=args.test_batches, callbacks=[summary]
    )
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
