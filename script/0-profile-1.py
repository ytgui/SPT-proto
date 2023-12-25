import math
import time
import torch
import argparse
from torch import nn, optim
from naive_gpt import layers
from fast_transformers import attention
from triton_flash import flash_attention
from torch import profiler


def load_layer(name: str,
               attention: str,
               seq_length: int,
               batch_size: int):
    n_heads = 32
    d_model = 2048
    cuda_device = 'cuda'

    #
    def loader():
        return torch.randn(
            [batch_size, seq_length, d_model],
            requires_grad=True, device=cuda_device
        )

    #
    if attention == 'full':
        model = layers.TransformerBlock(
            d_model=d_model, n_heads=n_heads,
            layernorm_fn=nn.LayerNorm(d_model),
            attention_fn=layers.VanillaAttention(
                d_head=d_model // n_heads,
                p_dropout=0.0
            ),
            feedforward_fn=nn.Identity(),
            attention_bias=True,
            pre_norm=True
        )
        return loader, model.to(cuda_device)
    else:
        raise RuntimeError


def profile(name: str,
            attention: str,
            seq_length: int,
            batch_size: int,
            compile: bool,
            d_lora: int):
    #
    print('name:', name)
    print('attention:', attention)
    print('seq_length:', seq_length)
    print('batch_size:', batch_size)
    print('compile:', compile)

    # model
    loader, model = load_layer(
        name=name,
        attention=attention,
        seq_length=seq_length,
        batch_size=batch_size
    )
    device = loader().device
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4,
        weight_decay=1e-2
    )

    # compile
    if compile:
        model = torch.compile(model)

    # warm up
    for _ in range(20):
        torch.cuda.synchronize()
        x = loader()
        y_1 = model(x)
        torch.sum(y_1).backward()
        optimizer.step()
        model.zero_grad()

    # simple
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    x = loader()
    y_1 = model(x)
    torch.sum(y_1).backward()
    optimizer.step()
    model.zero_grad()
    torch.cuda.synchronize()
    print('simple timing: {:.2f}ms'.format(
        1000.0 * (time.time() - before)
    ))

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
            torch.sum(y_1).backward()
            optimizer.step()
            model.zero_grad()
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
        '--name', default='opt-2048',
        help='specify model name or path'
    )
    parser.add_argument(
        '--attention', default='full',
        help='specify full, flash, lhs, pq'
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
    parser.add_argument(
        '--d_lora', help='dim oflow rank adaptation',
        default=16
    )
    args = parser.parse_args()

    #
    profile(
        name=args.name,
        attention=args.attention,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        compile=args.compile,
        d_lora=args.d_lora
    )


if __name__ == '__main__':
    main()


def legacy_main():
    n_heads = 32
    d_model = 2048
    batch_size = 16
    seq_length = 512
    d_head = d_model // n_heads
    cuda_device = 'cuda'

    #
    q = torch.randn(
        size=[batch_size, n_heads, seq_length, d_head],
        requires_grad=True, device=cuda_device
    )
    k = torch.randn(
        size=[batch_size, n_heads, seq_length, d_head],
        requires_grad=True, device=cuda_device
    )
    v = torch.randn(
        size=[batch_size, n_heads, seq_length, d_head],
        requires_grad=True, device=cuda_device
    )

    # N, L, H, E
    attention_fn = attention.ReformerAttention()

    # N, H, L, E
    y = flash_attention(q, k, v, 1.0 / math.sqrt(d_head))
    torch.sum(y).backward()

    #


if __name__ == '__main__':
    main()
