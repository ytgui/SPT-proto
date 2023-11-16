import torch
from torch import nn
from naive_gpt import kernels


class PQBase(nn.Module):
    def __init__(self,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int,
                 method: str):
        nn.Module.__init__(self)
        #
        self.method = method
        self.d_codeword = d_codeword
        self.n_codewords = n_codewords
        self.n_subspaces = n_subspaces
        #
        self.weight = nn.Parameter(
            torch.randn(
                [n_subspaces, n_codewords, d_codeword]
            )
        )
        self.loss_fn = nn.MSELoss()

    def forward(self,
                mode: str,
                z: torch.Tensor):
        assert mode in [
            'train', 'encode',
            'decode', 'quantize'
        ]
        assert z.dim() > 1
        if mode == 'decode':
            assert z.size(-1) == self.n_subspaces
        else:
            assert z.size(-1) == \
                self.d_codeword * self.n_subspaces

        # z_shape
        z_shape = list(z.size())[:-1] + [-1]

        # z_flat
        z_flat = z.flatten(end_dim=-2)
        z_flat = z_flat.view(
            [z_flat.size(0), self.n_subspaces, -1]
        )
        z_flat = z_flat.transpose(0, 1).contiguous()

        # indices
        if mode == 'decode':
            indices = z_flat
        else:
            if self.method == 'v1':
                dtype = z_flat.dtype
                # cdist only has float32
                distance = torch.cdist(
                    z_flat.type(torch.float),
                    self.weight.type(torch.float), p=1.0
                ).type(dtype)
                indices = torch.argmin(
                    distance, dim=-1, keepdim=True
                )
            elif self.method == 'v2':
                distance, indices = kernels.cdist(
                    z_flat, table=self.weight
                )
                indices = indices.unsqueeze(-1)
            else:
                raise RuntimeError
        assert indices.dim() == 3

        # encode
        if mode == 'encode':
            indices = indices.transpose(0, 1)
            indices = indices.view(z_shape)
            return indices.contiguous()

        # z_q: centroids
        indices = indices.type(torch.long)
        z_q_flat = torch.gather(
            self.weight, dim=1, index=indices.expand(
                size=[-1, -1, self.d_codeword]
            )
        )
        assert z_q_flat.dim() == 3
        z_q = torch.reshape(
            z_q_flat.transpose(0, 1), shape=z_shape
        )
        if mode in ['decode', 'quantize']:
            return z_q

        # training
        if mode != 'train':
            raise RuntimeError

        distance = torch.clamp(
            distance, min=1e-5
        )
        attention = torch.softmax(
            -torch.log(distance), dim=-1
        )
        z_w = torch.matmul(attention, self.weight)
        # minimize soft and hard centroids
        loss_w = self.loss_fn(z_w, target=z_q_flat)
        # minimize input to hard centroids
        loss_q = self.loss_fn(z_flat, target=z_q_flat)
        loss = loss_w + loss_q

        #
        return z_q, loss


class PQV1(PQBase):
    def __init__(self,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int):
        PQBase.__init__(
            self,
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces,
            method='v1'
        )


class PQV2(PQBase):
    def __init__(self,
                 d_codeword: int,
                 n_codewords: int,
                 n_subspaces: int):
        PQBase.__init__(
            self,
            d_codeword=d_codeword,
            n_codewords=n_codewords,
            n_subspaces=n_subspaces,
            method='v2'
        )
