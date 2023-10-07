import time
import torch
import random
from torch import nn
from torch import profiler
from naive_gpt import ext


class RoutedLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_size: int,
                 bias: bool = True):
        nn.Linear.__init__(
            self, in_features=in_features,
            out_features=out_features, bias=bias
        )
        #
        self.block_size = block_size
        assert in_features % block_size == 0
        assert out_features % block_size == 0
        self.in_blocks = in_features // block_size
        self.out_blocks = out_features // block_size
        self.router = nn.Sequential(
            nn.Linear(block_size, self.out_blocks),
            nn.Softmax(dim=-1)
        )

    @staticmethod
    def from_pretrained(block_size: int,
                        source: nn.Linear):
        model = RoutedLinear(
            in_features=source.in_features,
            out_features=source.out_features,
            bias=source.bias is not None,
            block_size=block_size
        )
        missing = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(missing.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view(
            [-1, self.in_blocks, self.block_size]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob.cpu(), k=1, dim=-1, sorted=False
        )

        #
        with torch.no_grad():
            y = ext.bspmm_forward_cuda(
                x, self.weight, topk.indices
            )
        y = y.view(
            list(x_size)[:-1] + [self.out_features]
        )
        return y


def test_moe():
    block_size = 4
    in_features = 2 * block_size
    out_features = 4 * block_size
    batch_size = 16
    cuda_device = 'cuda'

    #
    fc_1 = nn.Linear(
        in_features, out_features
    ).to(cuda_device)
    fc_2 = RoutedLinear.from_pretrained(
        block_size=block_size, source=fc_1
    ).to(cuda_device)
    x = torch.randn(
        [batch_size, in_features],
        device=cuda_device
    )

    # masked
    prob = fc_2.router(
        x.view([batch_size, -1, block_size])
    )
    indices = torch.argmax(prob, dim=-1)
    mask = nn.functional.one_hot(
        indices, num_classes=fc_2.out_blocks
    )
    mask = torch.repeat_interleave(
        mask, repeats=block_size, dim=-1
    )
    mask = torch.repeat_interleave(
        mask, repeats=block_size, dim=-2
    )
    masked_weight = torch.multiply(
        mask, fc_1.weight.T.unsqueeze(dim=0)
    )
    y_1 = torch.matmul(
        x.unsqueeze(dim=-2), masked_weight
    ).squeeze()

    # custom
    y_2 = fc_2(x)

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)

    #
    print('[PASS] test_moe()')


def bench_moe():
    block_size = 128
    in_features = 512
    out_features = 2048
    seq_length = 512
    batch_size = 16
    cuda_device = 'cuda'

    #
    fc_1 = nn.Linear(
        in_features, out_features
    ).to(cuda_device)
    fc_2 = RoutedLinear.from_pretrained(
        block_size=block_size, source=fc_1
    ).to(cuda_device)
    x = torch.randn(
        [batch_size, seq_length, in_features]
    ).to(cuda_device)

    # pre-warm
    # fc_1(x), fc_2(x)

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
            y_1 = fc_1(x)
    print(
        prof.key_averages().table(
            sort_by='cpu_time_total', row_limit=5
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
            y_2 = fc_2(x)
    print(
        prof.key_averages().table(
            sort_by='cpu_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_moe()')


def main():
    test_moe()
    bench_moe()


if __name__ == '__main__':
    main()
