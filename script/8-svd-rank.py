import time
import torch
import argparse
from torch import nn
from naive_gpt import layers, models


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

    # find sigma
    def find_sigma(table: list, ratio: float):
        for i, v in enumerate(table):
            if v < ratio:
                continue
            print('{:.2f} sigma: {}'.format(ratio, i))
            break

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

    #
    return


def evaluate(ckpt_path: str):
    ckpt = torch.load(f=ckpt_path)
    if ckpt_path.find('opt') > 0:
        model = models.OPTModel(**ckpt['config'])
    elif ckpt_path.find('llama') > 0:
        model = models.LLaMAModel(**ckpt['config'])
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['state_dict'])

    # ffn
    # svd_last_ffn(model=model)

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
