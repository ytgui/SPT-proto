import time
import torch
from torch import nn, profiler
from naive_gpt import kernels, layers


class SparseAttention(layers.VanillaAttention):
    def __init__(self,
                 d_head: int,
                 d_codeword: int,
                 n_codewords: int,
                 p_dropout: float):
        layers.VanillaAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.quantizer = layers.PQ(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        assert attn_mask is None
        assert q.size() == k.size()
        seq_length = q.size(1)

        # transpose: 0.4ms
        torch.cuda.synchronize()
        before = time.time()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        q = q.view([-1, q.size(-2), q.size(-1)])
        k = k.view([-1, k.size(-2), k.size(-1)])
        torch.cuda.synchronize()
        print('timing transpose: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))

        # quantizer: 2.3ms
        torch.cuda.synchronize()
        before = time.time()
        q_c = self.quantizer('encode', z=q)
        k_c = self.quantizer('encode', z=k)
        torch.cuda.synchronize()
        print('timing quantizer: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))

        # lookup: 2.0ms
        torch.cuda.synchronize()
        before = time.time()
        topk_indices = kernels.lookup(q_c, k_c, sparsity=8)
        csr_indices = topk_indices.flatten(start_dim=1)
        torch.cuda.synchronize()
        print('timing lookup: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))

        # indptr: 1.0ms
        torch.cuda.synchronize()
        before = time.time()
        top_k = seq_length // 8
        fixed_indptr = torch.arange(
            0, top_k * seq_length + 1, step=top_k,
            dtype=torch.int, device=q.device
        )
        fixed_indptr = fixed_indptr.view(1, -1).repeat(
            repeats=[csr_indices.size(0), 1]
        ).contiguous()
        torch.cuda.synchronize()
        print('timing indptr: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))

        # attention: 0.01s
        torch.cuda.synchronize()
        before = time.time()
        attn_values = kernels.sddmm(
            fixed_indptr, csr_indices, query=q, key=k
        )
        attn_values = kernels.softmax(
            fixed_indptr, csr_indices, values=attn_values
        )
        torch.cuda.synchronize()
        print('timing sddmm + softmax: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))
        return fixed_indptr, csr_indices, attn_values

    def _apply_attn(self,
                    attn: torch.Tensor,
                    v: torch.Tensor):
        torch.cuda.synchronize()
        before = time.time()
        indptr, indices, values = attn
        v = v.transpose(1, 2).contiguous()
        v = v.view([-1, v.size(-2), v.size(-1)])
        y = kernels.spmm(indptr, indices, values, v)
        torch.cuda.synchronize()
        print('timing transpose + spmm: {:.1f}ms'.format(
            1000.0 * (time.time() - before)
        ))
        return y


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
    dense_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    dense_fn = dense_fn.to(cuda_device)
    sparse_fn = SparseAttention(
        d_head=d_head, d_codeword=8,
        n_codewords=16, p_dropout=0.0
    )
    sparse_fn = sparse_fn.to(cuda_device)

    # warm up
    q, k, v = get_input()
    dense_fn(q, k, v, attn_mask=None)
    sparse_fn(q, k, v, attn_mask=None)

    # dense
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(1):
        q, k, v = get_input()
        dense_fn(q, k, v, attn_mask=None)
    torch.cuda.synchronize()
    print('timing dense: {:.1f}ms'.format(
        1000.0 * (time.time() - before)
    ))

    # sparse
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(1):
        q, k, v = get_input()
        sparse_fn(q, k, v, attn_mask=None)
    torch.cuda.synchronize()
    print('timing sparse: {:.1f}ms'.format(
        1000.0 * (time.time() - before)
    ))

    #
    return


if __name__ == '__main__':
    main()
