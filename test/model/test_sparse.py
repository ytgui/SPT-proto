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
        self.table = layers.PQTable(
            quantizer=self.quantizer, dim=1
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        assert attn_mask is None
        assert q.size() == k.size()
        seq_length = q.size(1)

        # loss: 0.01s
        q_c, loss_q = self.quantizer('train', z=q)
        k_c, loss_k = self.quantizer('train', z=k)
        self.register_buffer(
            'loss', loss_q + loss_k, persistent=False
        )

        # distance: 0.15s
        distance = self.table(q_c, k_c)
        distance = torch.transpose(distance, 1, 2)
        

        # indptr and indices: 0.01s
        top_k = seq_length // 8
        top_indices = torch.topk(
            distance, k=top_k, largest=False, sorted=False
        )[-1]
        csr_indices = top_indices.flatten(start_dim=2)
        csr_indices = csr_indices.flatten(end_dim=1)
        csr_indices = csr_indices.type(torch.int)
        fixed_indptr = torch.arange(
            0, top_k * seq_length + 1, step=top_k,
            dtype=torch.int, device=q.device
        )
        fixed_indptr = fixed_indptr.view(1, -1).repeat(
            repeats=[csr_indices.size(0), 1]
        ).contiguous()

        # attention: 0.01s
        query = q.transpose(1, 2).contiguous().flatten(end_dim=1)
        key = k.transpose(1, 2).contiguous().flatten(end_dim=1)
        attn_values = kernels.sddmm(
            fixed_indptr, csr_indices, query=query, key=key
        )
        attn_values = kernels.softmax(
            fixed_indptr, csr_indices, values=attn_values
        )
        return fixed_indptr, csr_indices, attn_values

    def _apply_attn(self,
                    attn: torch.Tensor,
                    v: torch.Tensor):
        indptr, indices, attn_values = attn
        v = v.transpose(1, 2).contiguous().flatten(end_dim=1)
        return kernels.spmm(indptr, indices, attn_values, v)


def main():
    d_head = 64
    n_heads = 8
    batch_size = 16
    seq_length = 512
    cuda_device = 'cuda'

    #
    x = torch.randn(
        [batch_size, seq_length,
         n_heads, d_head],
        device=cuda_device
    )
    dense_fn = layers.VanillaAttention(
        d_head=d_head, p_dropout=0.0
    )
    dense_fn = dense_fn.to(cuda_device)
    sparse_fn = SparseAttention(
        d_head=d_head, d_codeword=4,
        n_codewords=64, p_dropout=0.0
    )
    sparse_fn = sparse_fn.to(cuda_device)

    #
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        y_1 = dense_fn(x, x, x, attn_mask=None)
    torch.cuda.synchronize()
    print('timing dense:', time.time() - before)

    #
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        y_2 = sparse_fn(x, x, x, attn_mask=None)
    torch.cuda.synchronize()
    print('timing sparse:', time.time() - before)

    #
    return


if __name__ == '__main__':
    main()
