import time
import torch
from torch import nn, profiler
from naive_gpt import layers


def bench_mha():
    d_head = 64
    n_heads = 8
    batch_size = 16
    seq_length = 512
    cuda_device = 'cuda'

    #
    q = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    k = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    v = torch.randn(
        [batch_size, seq_length, n_heads, d_head],
        requires_grad=True, device=cuda_device
    )
    origin_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    origin_fn = origin_fn.to(cuda_device)
    compiled_fn = torch.compile(origin_fn)

    # warm up
    y_1 = origin_fn(q, k, v, attn_mask=None)
    y_2 = compiled_fn(q, k, v, attn_mask=None)
    torch.sum(y_1).backward()
    torch.sum(y_2).backward()

    #
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
            y_1 = origin_fn(q, k, v, attn_mask=None)
            torch.sum(y_1).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
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
            y_2 = compiled_fn(q, k, v, attn_mask=None)
            torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_mha()')


def bench_ffn():
    d_model = 512
    batch_size = 16
    seq_length = 512
    d_feedforward = 4 * d_model
    cuda_device = 'cuda'

    #
    x = torch.randn(
        [batch_size, seq_length, d_model],
        requires_grad=True, device=cuda_device
    )
    origin_fn = layers.Feedforward(
        d_model, d_feedforward=d_feedforward,
        p_dropout=0.0, activation=nn.ReLU()
    )
    origin_fn = origin_fn.to(cuda_device)
    compiled_fn = torch.compile(origin_fn)

    # warm up
    y_1 = origin_fn(x)
    y_2 = compiled_fn(x)
    torch.sum(y_1).backward()
    torch.sum(y_2).backward()

    #
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
            y_1 = origin_fn(x)
            torch.sum(y_1).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
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
            y_2 = compiled_fn(x)
            torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_ffn()')


def main():
    bench_mha()
    bench_ffn()


if __name__ == '__main__':
    main()
