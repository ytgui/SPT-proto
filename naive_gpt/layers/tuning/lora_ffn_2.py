import torch
from torch import nn
from naive_gpt import layers


class LoRARoutedFFN(layers.RoutedFFN):
    def __init__(self,
                 d_lora: int,
                 block_size: int,
                 d_model: int,
                 d_feedforward: int,
                 activation: nn.Module):
        layers.RoutedFFN.__init__(
            self,
            block_size=block_size,
            d_model=d_model,
            d_feedforward=d_feedforward,
            activation=activation,
            p_dropout=0.0
        )
        # LoRA
        self.fc1 = layers.LoRALinear(
            d_lora=d_lora, in_features=d_model,
            out_features=d_feedforward
        )
        self.fc2 = layers.LoRALinear(
            d_lora=d_lora, in_features=d_feedforward,
            out_features=d_model
        )

    @staticmethod
    def from_pretrained(d_lora: int,
                        block_size: int,
                        source: layers.Feedforward):
        assert isinstance(
            source, layers.Feedforward
        )
        model = LoRARoutedFFN(
            d_lora=d_lora,
            block_size=block_size,
            d_model=source.d_model,
            d_feedforward=source.d_feedforward,
            activation=source.activation
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def _apply_ffn(self,
                   x: torch.Tensor,
                   bias_1: torch.Tensor,
                   weight_1: torch.Tensor,
                   weight_2: torch.Tensor):
        # route
        x_size = x.size()
        x = x.view(
            [-1, self.d_model]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 4,
            dim=-1, sorted=False
        )
        indices = topk.indices

        # mask
        masks = []
        for i in range(self.n_blocks):
            cmp = torch.eq(indices, i)
            masks.append(
                torch.sum(
                    cmp, dim=-1, dtype=torch.bool
                )
            )

        # fc1
        h = torch.matmul(
            torch.matmul(
                x, self.fc1.lora.left.weight
            ),
            self.fc1.lora.right.weight.T
        )
        h = h.view(
            [-1, self.n_blocks, self.block_size]
        )
        for i in range(self.n_blocks):
            x_i = x[masks[i]]
            b_i, w_i = bias_1[i], weight_1[i]
            h[masks[i], i] += torch.addmm(
                b_i, x_i, w_i.T, beta=1.0, alpha=1.0
            )
        h = self.activation(h)

        # fc2
        y = torch.matmul(
            torch.matmul(
                h.view([h.size(0), -1]),
            self.fc2.lora.left.weight
            ),
            self.fc2.lora.right.weight.T
        )
        for i in range(self.n_blocks):
            y[masks[i]] += torch.matmul(
                h[masks[i], i], weight_2[i]
            )
        y += self.fc2.bias.view([1, -1])

        #
        return y.view(x_size)

    def forward(self, x: torch.Tensor):
        bias_1 = self.fc1.bias.view(
            [self.n_blocks, self.block_size]
        )
        weight_1 = self.fc1.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_2 = self.fc2.weight.view(
            [-1, self.n_blocks, self.block_size]
        )
        weight_2 = torch.permute(
            weight_2, dims=[1, 2, 0]
        )
        return self._apply_ffn(
            x, bias_1=bias_1, weight_1=weight_1,
            weight_2=weight_2.contiguous()
        )


class LoRARoutedLLaMaFFN(layers.RoutedLLaMaFFN):
    def __init__(self,
                 d_lora: int,
                 block_size: int,
                 d_model: int,
                 d_feedforward: int,
                 activation: nn.Module):
        layers.RoutedLLaMaFFN.__init__(
            self, d_model, d_feedforward,
            block_size=block_size, activation=activation
        )
        # LoRA
        self.gate = layers.LoRALinear(
            d_lora=d_lora, in_features=d_model,
            out_features=d_feedforward, bias=False
        )
        self.side = layers.LoRALinear(
            d_lora=d_lora, in_features=d_model,
            out_features=d_feedforward, bias=False
        )
        self.down = layers.LoRALinear(
            d_lora=d_lora, in_features=d_feedforward,
            out_features=d_model, bias=False
        )

    @staticmethod
    def from_pretrained(d_lora: int,
                        block_size: int,
                        source: layers.LLaMaFeedforward):
        assert isinstance(
            source, layers.LLaMaFeedforward
        )
        model = LoRARoutedLLaMaFFN(
            d_lora=d_lora,
            block_size=block_size,
            d_model=source.d_model,
            d_feedforward=source.d_feedforward,
            activation=source.activation
        )
        output = model.load_state_dict(
            source.state_dict(), strict=False
        )
        if len(output.missing_keys) != 2:
            raise RuntimeError
        return model

    def forward(self, x: torch.Tensor):
        # lora
        weight_gate = torch.matmul(
            self.gate.lora.right.weight,
            self.gate.lora.left.weight.T
        ) + self.gate.weight
        weight_side = torch.matmul(
            self.side.lora.right.weight,
            self.side.lora.left.weight.T
        ) + self.side.weight
        weight_down = torch.matmul(
            self.down.lora.right.weight,
            self.down.lora.left.weight.T
        ) + self.down.weight

        #
        weight_gate = weight_gate.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_side = weight_side.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_down = weight_down.view(
            [-1, self.n_blocks, self.block_size]
        )
        weight_down = torch.permute(
            weight_down, dims=[1, 2, 0]
        )
        return self._apply_ffn(
            x, weight_gate=weight_gate,
            weight_side=weight_side,
            weight_down=weight_down.contiguous()
        )
