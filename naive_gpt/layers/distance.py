import torch
from torch import nn
from .product import PQ


class PQTable(nn.Module):
    def __init__(self,
                 quantizer: PQ,
                 dim: int = 0):
        nn.Module.__init__(self)
        #
        self.dim = dim
        self.quantizer = quantizer
        self.n_subspaces = quantizer.n_subspaces
        #
        self.rebuild_table()

    def rebuild_table(self):
        # copy codewords
        weight = torch.clone(
            self.quantizer.weight.detach()
        )
        self.table: torch.Tensor
        self.register_buffer(
            'table', torch.cdist(weight, weight, p=2.0)
        )

    def forward(self,
                q_code: torch.Tensor,
                k_code: torch.Tensor):
        assert q_code.size(-1) == self.n_subspaces
        assert k_code.size(-1) == self.n_subspaces

        #
        q_chunks = torch.chunk(
            q_code, chunks=self.n_subspaces, dim=-1
        )
        k_chunks = torch.chunk(
            k_code, chunks=self.n_subspaces, dim=-1
        )

        #
        distance = []
        for i in range(self.n_subspaces):
            table = self.table[i]
            q, k = q_chunks[i], k_chunks[i]
            d = table[q, k.transpose(-1, self.dim)]
            distance.append(d)
        distance = torch.stack(distance, dim=-1)
        distance = torch.sum(distance, dim=-1)
        return distance
