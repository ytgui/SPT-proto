import torch
from naive_gpt import kernels, layers


class SparseVanillaAttentionV1(layers.VanillaAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int):
        layers.VanillaAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.d_codeword = d_codeword
        self.n_codewords = n_codewords
        self.n_subspaces = n_subspaces
        #
        self.quantizer = layers.PQV1(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces
        )
        #
        self.trigger: torch.Tensor
        self.register_buffer(
            'trigger', torch.scalar_tensor(False, dtype=torch.bool)
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        loss_q = self.quantizer('train', z=q)[-1]
        loss_k = self.quantizer('train', z=k)[-1]
        self.register_buffer(
            'loss', loss_q + loss_k, persistent=False
        )
        return layers.VanillaAttention._get_attn(
            self, q, k, attn_mask=attn_mask
        )


class SparseVanillaAttentionV2(layers.VanillaAttention):
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
        self.quantizer = layers.PQV2(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
        )
        #
        self.trigger: torch.Tensor
        self.register_buffer(
            'trigger', torch.scalar_tensor(False, dtype=torch.bool)
        )

    @staticmethod
    def from_pretrained(source: SparseVanillaAttentionV1):
        assert isinstance(
            source, SparseVanillaAttentionV1
        )
        model = layers.SparseVanillaAttentionV2(
            d_head=source.d_head, d_codeword=source.d_codeword,
            n_codewords=source.n_codewords, p_dropout=0.0
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 0:
            raise RuntimeError
        return model

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        assert q.size() == k.size()
        seq_length = q.size(1)

        # transpose
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        q = q.view([-1, q.size(-2), q.size(-1)])
        k = k.view([-1, k.size(-2), k.size(-1)])

        # quantizer
        if self.trigger.is_nonzero():
            self.trigger.logical_not_()
            loss_q = self.quantizer('train', z=q)[-1]
            loss_k = self.quantizer('train', z=k)[-1]
            self.register_buffer(
                'loss', loss_q + loss_k, persistent=False
            )
        q_c = self.quantizer('encode', z=q)
        k_c = self.quantizer('encode', z=k)

        # lookup
        topk_indices = kernels.lookup(
            q_c, k_c, sparse_coeff=8
        )
        csr_indices = topk_indices.flatten(start_dim=1)

        # indptr
        top_k = seq_length // 8
        fixed_indptr = torch.arange(
            0, top_k * seq_length + 1, step=top_k,
            dtype=torch.int, device=q.device
        )

        # attention
        attn_values = kernels.sddmm(
            fixed_indptr, csr_indices, query=q, key=k
        )
        attn_values = torch.clamp_(
            self.scaling * attn_values, min=-10.0, max=10.0
        )
        attn_values = kernels.softmax(
            fixed_indptr, csr_indices, values=attn_values
        )
        return fixed_indptr, csr_indices, attn_values

    def _apply_attn(self,
                    attn: torch.Tensor,
                    v: torch.Tensor):
        v_size = v.size()
        indptr, indices, values = attn
        v = v.transpose(1, 2).contiguous()
        v = v.view([-1, v.size(-2), v.size(-1)])
        y = kernels.spmm(indptr, indices, values, v)
        y = y.transpose(1, 2).contiguous()
        return y.view(v_size)


class SparseRotaryAttentionV1(layers.RotaryAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int):
        layers.RotaryAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.d_codeword = d_codeword
        self.n_codewords = n_codewords
        self.n_subspaces = n_subspaces
        #
        self.quantizer = layers.PQV1(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
        )
        #
        self.trigger: torch.Tensor
        self.register_buffer(
            'trigger', torch.scalar_tensor(False, dtype=torch.bool)
        )

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        # rotary
        q = self.embedding(
            q, ids=self.cached_ids[:q.size(1)]
        )
        k = self.embedding(
            k, ids=self.cached_ids[:k.size(1)]
        )

        # quantize
        loss_q = self.quantizer('train', z=q)[-1]
        loss_k = self.quantizer('train', z=k)[-1]
        self.register_buffer(
            'loss', loss_q + loss_k, persistent=False
        )
        return layers.VanillaAttention._get_attn(
            self, q, k, attn_mask=attn_mask
        )


class SparseRotaryAttentionV2(layers.RotaryAttention):
    def __init__(self,
                 d_head: int,
                 p_dropout: float,
                 d_codeword: int,
                 n_codewords: int):
        layers.RotaryAttention.__init__(
            self, d_head=d_head,
            p_dropout=p_dropout
        )
        #
        self.quantizer = layers.PQV2(
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=d_head // d_codeword
        )
        #
        self.trigger: torch.Tensor
        self.register_buffer(
            'trigger', torch.scalar_tensor(False, dtype=torch.bool)
        )

    @staticmethod
    def from_pretrained(source: SparseRotaryAttentionV1):
        assert isinstance(
            source, SparseRotaryAttentionV1
        )
        model = layers.SparseRotaryAttentionV2(
            d_head=source.d_head, d_codeword=source.d_codeword,
            n_codewords=source.n_codewords, p_dropout=0.0
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 0:
            raise RuntimeError
        return model

    def _get_attn(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  attn_mask: torch.Tensor):
        assert q.size() == k.size()
        seq_length = q.size(1)

        # rotary
        q = self.embedding(
            q, ids=self.cached_ids[:q.size(1)]
        )
        k = self.embedding(
            k, ids=self.cached_ids[:k.size(1)]
        )

        # transpose
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        q = q.view([-1, q.size(-2), q.size(-1)])
        k = k.view([-1, k.size(-2), k.size(-1)])

        # quantizer
        if self.trigger.is_nonzero():
            self.trigger.logical_not_()
            loss_q = self.quantizer('train', z=q)[-1]
            loss_k = self.quantizer('train', z=k)[-1]
            self.register_buffer(
                'loss', loss_q + loss_k, persistent=False
            )
        q_c = self.quantizer('encode', z=q)
        k_c = self.quantizer('encode', z=k)

        # lookup
        topk_indices = kernels.lookup(
            q_c, k_c, sparse_coeff=8
        )
        csr_indices = topk_indices.flatten(start_dim=1)

        # indptr
        top_k = seq_length // 8
        fixed_indptr = torch.arange(
            0, top_k * seq_length + 1, step=top_k,
            dtype=torch.int, device=q.device
        )

        # attention
        attn_values = kernels.sddmm(
            fixed_indptr, csr_indices, query=q, key=k
        )
        attn_values = torch.clamp_(
            self.scaling * attn_values, min=-10.0, max=10.0
        )
        attn_values = kernels.softmax(
            fixed_indptr, csr_indices, values=attn_values
        )
        return fixed_indptr, csr_indices, attn_values

    def _apply_attn(self,
                    attn: torch.Tensor,
                    v: torch.Tensor):
        v_size = v.size()
        indptr, indices, values = attn
        v = v.transpose(1, 2).contiguous()
        v = v.view([-1, v.size(-2), v.size(-1)])
        y = kernels.spmm(indptr, indices, values, v)
        y = y.transpose(1, 2).contiguous()
        return y.view(v_size)
