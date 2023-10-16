import torch
import argparse
from naive_gpt import models, utils


def show_info(ckpt_path: str, tuning: str, d_lora: int):
    print('ckpt_path:', ckpt_path)
    print('tuning:', tuning)
    print('d_lora:', d_lora)

    # model
    ckpt = torch.load(f=ckpt_path)
    if ckpt_path.find('opt') > 0:
        model = models.OPTModel(**ckpt['config'])
    elif ckpt_path.find('llama') > 0:
        model = models.LLaMAModel(**ckpt['config'])
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # upgrade
    if tuning == 'full':
        pass
    elif tuning == 'lora':
        upgrader = utils.ModuleUpgrader(
            handler=utils.LoRAHandler(
                lora_r=d_lora,
                lora_dropout=0.0
            )
        )
        model = upgrader.visit(model)
    elif tuning == 'sparse':
        upgrader = utils.ModuleUpgrader(
            handler=utils.SparseLoRAHandler(
                lora_r=d_lora,
                lora_dropout=0.0,
                stage=1
            )
        )
        model = upgrader.visit(model)
    else:
        raise RuntimeError

    # parameters
    trainable = 0
    non_trainable = 0
    for param in model.parameters():
        param: torch.Tensor
        if param.requires_grad:
            trainable += param.numel()
        else:
            non_trainable += param.numel()
    print('trainable: {:.2f}M'.format(trainable / (2 ** 20)))
    print('non_trainable: {:.2f}M'.format(non_trainable / (2 ** 20)))

    #
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', help='specify model path',
        default='.data/opt-1.3b.ckpt'
    )
    parser.add_argument(
        '--tuning', default='sparse',
        help='specify full, lora, or sparse'
    )
    parser.add_argument(
        '--d_lora', help='dim oflow rank adaptation',
        default=16
    )
    args = parser.parse_args()
    show_info(ckpt_path=args.ckpt, tuning=args.tuning, d_lora=args.d_lora)


if __name__ == '__main__':
    main()
