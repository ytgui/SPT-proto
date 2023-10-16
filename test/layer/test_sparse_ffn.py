import time
import torch
from torch import nn, profiler
from naive_gpt import layers


class NaiveRoutedFFN(layers.RoutedFFN):
    def forward(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view([-1, self.d_model])

        # topk
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 4,
            dim=-1, sorted=False
        )

        # mask
        mask = torch.zeros(
            [x.size(0), self.n_blocks]
        )
        mask = torch.scatter(
            torch.zeros_like(mask),
            dim=-1, index=topk.indices,
            src=torch.ones_like(topk.values)
        )
        mask = torch.repeat_interleave(
            mask, repeats=self.block_size, dim=-1
        )

        # fc
        h = torch.multiply(
            mask, self.fc1(x)
        )
        h = self.activation(h)
        y = self.fc2(h)
        return y.view(x_size)


def test_routed_ffn():
    d_model = 8
    d_feedforward = 64
    block_size = 4
    seq_length = 16
    batch_size = 4

    #
    x = torch.randn(
        [batch_size, seq_length, d_model],
        requires_grad=True
    )
    ffn_1 = layers.RoutedFFN(
        d_model=d_model,
        d_feedforward=d_feedforward,
        block_size=block_size,
        activation=nn.ReLU()
    )
    ffn_2 = NaiveRoutedFFN(
        d_model=d_model,
        d_feedforward=d_feedforward,
        block_size=block_size,
        activation=nn.ReLU()
    )
    ffn_2.load_state_dict(
        state_dict=ffn_1.state_dict()
    )

    #
    y_1 = ffn_1(x)
    torch.sum(y_1).backward()
    grad_x_1 = x.grad.detach().clone()
    grad_b1_1 = torch.clone(
        ffn_1.fc1.bias.grad.detach()
    )
    grad_b2_1 = torch.clone(
        ffn_1.fc2.bias.grad.detach()
    )
    grad_w1_1 = torch.clone(
        ffn_1.fc1.weight.grad.detach()
    )
    grad_w2_1 = torch.clone(
        ffn_1.fc2.weight.grad.detach()
    )

    #
    x.grad.zero_()
    y_2 = ffn_2(x)
    torch.sum(y_2).backward()
    grad_b1_2 = torch.clone(
        ffn_2.fc1.bias.grad.detach()
    )
    grad_b2_2 = torch.clone(
        ffn_2.fc2.bias.grad.detach()
    )
    grad_w1_2 = torch.clone(
        ffn_2.fc1.weight.grad.detach()
    )
    grad_w2_2 = torch.clone(
        ffn_2.fc2.weight.grad.detach()
    )
    grad_x_2 = x.grad.detach().clone()

    # check
    assert torch.allclose(y_1, y_2, atol=1e-3)
    assert torch.allclose(grad_x_1, grad_x_2, atol=1e-3)
    assert torch.allclose(grad_b1_1, grad_b1_2, atol=1e-3)
    assert torch.allclose(grad_b2_1, grad_b2_2, atol=1e-3)
    assert torch.allclose(grad_w1_1, grad_w1_2, atol=1e-3)
    assert torch.allclose(grad_w2_1, grad_w2_2, atol=1e-3)

    #
    print('[PASS] test_routed_ffn()')


def bench_routed_ffn():
    d_model = 2048
    d_feedforward = 8192
    block_size = 1024
    seq_length = 512
    batch_size = 16
    cuda_device = 'cuda'

    #
    ffn_1 = layers.Feedforward(
        d_model=d_model,
        d_feedforward=d_feedforward,
        activation=nn.ReLU(),
        p_dropout=0.0
    )
    ffn_2 = layers.RoutedFFN(
        d_model=d_model,
        d_feedforward=d_feedforward,
        block_size=block_size,
        activation=nn.ReLU()
    )
    ffn_1 = ffn_1.to(cuda_device)
    ffn_2 = ffn_2.to(cuda_device)
    x = torch.randn(
        [batch_size, seq_length, d_model],
        device=cuda_device
    )

    # pre-warm
    for _ in range(20):
        y_1, y_2 = ffn_1(x), ffn_2(x)
        torch.sum(y_1).backward()
        torch.sum(y_2).backward()
    torch.cuda.synchronize()

    # simple full
    torch.cuda.synchronize()
    before = time.time()
    y_1 = ffn_1(x)
    torch.cuda.synchronize()
    print('timing 0', 1000.0 * (time.time() - before))

    # simple routed
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    y_2 = ffn_2(x)
    torch.cuda.synchronize()
    print('timing 1', 1000.0 * (time.time() - before))

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
            y_1 = ffn_1(x)
            torch.sum(y_1).backward()
            torch.cuda.synchronize()
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
            y_2 = ffn_2(x)
            torch.sum(y_2).backward()
            torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total', row_limit=5
        )
    )

    #
    print('[PASS] bench_routed_ffn()')


def main():
    test_routed_ffn()
    bench_routed_ffn()


if __name__ == '__main__':
    main()
