import torch
from torch import nn


class FnModule(nn.Module):
    def __init__(self,
                 fn: callable,
                 *args, **kwargs):
        nn.Module.__init__(self)
        #
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.fn(
            *args, *self.args,
            **kwargs, **self.kwargs
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)
        return self.weight * x
