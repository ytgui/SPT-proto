import time
import torch
from torch import nn, profiler
from naive_gpt import layers


def test_ffn():
    block_size = 4
    in_features = 8
    out_features = 32
    batch_size = 16 * 256

    #
    fc_1 = layers.Feedforward(
        d_model=in_features,
        d_feedforward=out_features,
        activation=nn.ReLU(),
        p_dropout=0.0
    )
    fc_2 = layers.RoutedFFN(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        actication=nn.ReLU()
    )
    x = torch.randn(
        [batch_size, in_features]
    )


def bench_ffn():
    block_size = 256
    in_features = 512
    out_features = 2048
    batch_size = 16 * 512
    cuda_device = 'cuda'

    #
    fc_1 = layers.Feedforward(
        d_model=in_features,
        d_feedforward=out_features,
        activation=nn.ReLU(),
        p_dropout=0.0
    )
    fc_2 = layers.RoutedFFN(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        actication=nn.ReLU()
    )
    fc_1 = fc_1.to(cuda_device)
    fc_2 = fc_2.to(cuda_device)
    x = torch.randn(
        [batch_size, in_features],
        device=cuda_device
    )

    # pre-warm
    for _ in range(20):
        y_1, y_2 = fc_1(x), fc_2(x)
        torch.sum(y_1).backward()
        torch.sum(y_2).backward()

    # full
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
            y_1 = fc_1(x)
            # torch.sum(y_1).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    # routed
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
            y_2 = fc_2(x)
            # torch.sum(y_2).backward()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )


def main():
    test_ffn()
    bench_ffn()


if __name__ == '__main__':
    main()
