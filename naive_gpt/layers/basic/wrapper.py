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
