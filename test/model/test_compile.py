import time
import torch
from naive_gpt import layers


def main():
    d_head = 64
    n_heads = 4
    batch_size = 16
    seq_length = 1024
    cuda_device = 'cuda'

    #
    def get_input():
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
        return q, k, v

    #
    origin_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    origin_fn = origin_fn.to(cuda_device)
    compiled_fn = torch.compile(origin_fn)

    # warm up
    q, k, v = get_input()
    torch.sum(
        origin_fn(q, k, v, attn_mask=None)
    ).backward()
    torch.sum(
        compiled_fn(q, k, v, attn_mask=None)
    ).backward()

    #
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        q, k, v = get_input()
        y_1 = origin_fn(q, k, v, attn_mask=None)
        torch.sum(y_1).backward()
    torch.cuda.synchronize()
    print('timing original: {:.1f}ms'.format(
        1000.0 * (time.time() - before)
    ))

    #
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        q, k, v = get_input()
        y_2 = compiled_fn(q, k, v, attn_mask=None)
        torch.sum(y_2).backward()
    torch.cuda.synchronize()
    print('timing compiled: {:.1f}ms'.format(
        1000.0 * (time.time() - before)
    ))


if __name__ == '__main__':
    main()
