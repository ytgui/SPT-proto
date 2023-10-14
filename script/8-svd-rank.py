import os
import torch
import argparse
from torch import nn
from naive_gpt import loaders, layers, models


def find_sigma(table: list, ratio: float):
    for i, v in enumerate(table):
        if v < ratio:
            continue
        print('{:.2f} sigma: {}'.format(ratio, i))
        break


def svd_last_ffn(model: nn.Module):
    # feedforward
    last_ffd: nn.Module = None
    for module in model.modules():
        if not isinstance(
            module, (layers.Feedforward,
                     layers.LLaMaFeedforward)
        ):
            continue
        last_ffd = module

    # fc weights
    fc_weights = []
    for param in last_ffd.parameters():
        param: torch.Tensor
        if param.dim() == 1:
            continue
        fc_weights.append(param)

    # svd decompose
    for w in fc_weights:
        w: torch.Tensor
        print('w_size:', w.size())
        _, sigma, _ = torch.svd(w)
        print('sigma_size:', sigma.size())
        cum_sigma = torch.cumsum(sigma, dim=-1)
        cumsum_ratio = cum_sigma / torch.sum(sigma)
        cumsum_ratio = cumsum_ratio.tolist()
        print(
            'cumsum(sigma) / sum(sigma):', [
                '{}: {:.2f}'.format(i, v)
                for i, v in enumerate(cumsum_ratio)
            ]
        )
        find_sigma(table=cumsum_ratio, ratio=0.50)
        find_sigma(table=cumsum_ratio, ratio=0.75)
        find_sigma(table=cumsum_ratio, ratio=0.80)
        find_sigma(table=cumsum_ratio, ratio=0.90)
        find_sigma(table=cumsum_ratio, ratio=0.95)


def svd_ffn_hidden(model: nn.Module):
    # feedforward
    last_ffd: nn.Module = None
    for module in model.modules():
        if not isinstance(
            module, (layers.Feedforward,
                     layers.LLaMaFeedforward)
        ):
            continue
        last_ffd = module

    # callback
    def on_forward(module: nn.Linear,
                   args: list[torch.Tensor],
                   output: torch.Tensor):
        for x in [args[0][0], output[0]]:
            print('x_size:', x.size())
            _, sigma, _ = torch.svd(x)
            print('sigma_size:', sigma.size())
            cum_sigma = torch.cumsum(sigma, dim=-1)
            cumsum_ratio = cum_sigma / torch.sum(sigma)
            cumsum_ratio = cumsum_ratio.tolist()
            print(
                'cumsum(sigma) / sum(sigma):', [
                    '{}: {:.2f}'.format(i, v)
                    for i, v in enumerate(cumsum_ratio)
                ]
            )
            find_sigma(table=cumsum_ratio, ratio=0.50)
            find_sigma(table=cumsum_ratio, ratio=0.75)
            find_sigma(table=cumsum_ratio, ratio=0.80)
            find_sigma(table=cumsum_ratio, ratio=0.90)

    # linears
    for module in last_ffd.modules():
        module: nn.Module
        if not isinstance(module, nn.Linear):
            continue
        module.register_forward_hook(on_forward)

    # loader
    x = None
    dm = loaders.WikitextDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=512, batch_size=1,
        tokenizer='opt', num_workers=1
    )
    for batch in dm.train_dataloader():
        x = batch
        assert batch.size(0) == 1
        # skip non-full sequence
        if x[0][-1] != dm.pad_value:
            break

    # run once
    model(x)


@torch.no_grad()
def evaluate(ckpt_path: str):
    ckpt = torch.load(f=ckpt_path)
    if ckpt_path.find('opt') != -1:
        model = models.OPTModel(**ckpt['config'])
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # ffn
    svd_last_ffn(model=model)

    # hidden
    svd_ffn_hidden(model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', help='specify model path',
        default='.data/opt-2.7b.ckpt'
    )
    args = parser.parse_args()
    evaluate(ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
