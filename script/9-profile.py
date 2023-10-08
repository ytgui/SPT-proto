import time
import torch
import argparse
from torch import nn, profiler
from naive_gpt import layers


def load_model(name: str,
               module: str,
               seq_length: int,
               batch_size: int):
    cuda_device = 'cuda'

    #
    if name == 'facebook/opt-1.3b':
        n_heads = 32
        d_model = 2048
        d_feedforward = 4 * 2048

        #
        def loader():
            return torch.randn(
                [batch_size, seq_length, d_model],
                requires_grad=True, device=cuda_device
            )
        #
        if module == 'mha':
            model = layers.VanillaTransformerBlock(
                d_model=d_model, n_heads=n_heads,
                attention_fn=layers.VanillaAttention(
                    d_head=d_model // n_heads,
                    p_dropout=0.0
                ),
                feedforward_fn=nn.Identity(),
                pre_norm=True
            )
            return loader, model.to(cuda_device)
        elif module == 'ffn':
            model = layers.Feedforward(
                d_model=d_model,
                d_feedforward=d_feedforward,
                activation=nn.ReLU(),
                p_dropout=0.0
            )
            return loader, model.to(cuda_device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def profile(name: str,
            module: str,
            seq_length: int,
            batch_size: int,
            backward: bool,
            compile: bool):
    #
    print('model:', name, module)
    print('seq_length:', seq_length)
    print('batch_size:', batch_size)
    print('backward:', backward)
    print('compile:', compile)

    # model
    loader, model = load_model(
        name, module, seq_length, batch_size=batch_size
    )
    if compile:
        model = torch.compile(model)

    # warm up
    for _ in range(20):
        torch.cuda.synchronize()
        x = loader()
        y_1 = model(x)
        if not backward:
            continue
        torch.sum(y_1).backward()

    # profile
    time.sleep(2.0)
    with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_modules=True
    ) as prof:
        for _ in range(20):
            torch.cuda.synchronize()
            x = loader()
            y_1 = model(x)
            if not backward:
                continue
            torch.sum(y_1).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    # memory
    print(torch.cuda.memory_summary())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default='facebook/opt-1.3b',
        help='specify model name or path'
    )
    parser.add_argument(
        '--module', default='mha',
        help='specify module in mha or ffn'
    )
    parser.add_argument(
        '--backward', action='store_true',
        help='specify to enable backard'
    )
    parser.add_argument(
        '--compile', action='store_true',
        help='specify to enable torch.compile'
    )
    parser.add_argument(
        '--seq_length', default=512, type=int,
        help='specify sequence length'
    )
    parser.add_argument(
        '--batch_size', default=16, type=int,
        help='specify batch size'
    )
    args = parser.parse_args()

    #
    profile(
        name=args.name,
        module=args.module,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        backward=args.backward,
        compile=args.compile
    )


if __name__ == '__main__':
    main()
