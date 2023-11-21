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

    def forward(self, x: torch.Tensor):
        # route
        x_size = x.size()
        x = x.view(
            [-1, self.d_model]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 2,
            dim=-1, sorted=False
        )
        indices = topk.indices

        # blocking
        bias_1 = self.fc1.bias.view(
            [self.n_blocks, self.block_size]
        )
        weight_1 = self.fc1.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        lora_right_1 = self.fc1.lora.right.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_2 = self.fc2.weight.view(
            [-1, self.n_blocks, self.block_size]
        )
        weight_2 = torch.permute(
            weight_2, dims=[1, 2, 0]
        ).contiguous()
        lora_left_2 = self.fc2.lora.left.weight.view(
            [self.n_blocks, self.block_size, -1]
        )

        #
        y = torch.zeros_like(x)
        for i in range(self.n_blocks):
            cmp = torch.eq(indices, i)
            mask = torch.sum(
                cmp, dim=-1, dtype=torch.bool
            )
            coeff = 2.0 * prob[mask, i].unsqueeze(-1)
            # fc1
            x_i = x[mask]
            b_i, w_i = bias_1[i], weight_1[i]
            h = coeff * torch.addmm(
                b_i, x_i, w_i.T, beta=1.0, alpha=1.0
            )
            h += torch.matmul(
                torch.matmul(
                    x_i, self.fc1.lora.left.weight
                ),
                lora_right_1[i].T
            )
            h = self.activation(h)
            # fc2
            w_i = weight_2[i]
            y[mask] += torch.matmul(
                torch.matmul(h, lora_left_2[i]),
                self.fc2.lora.right.weight.T
            ) + coeff * torch.matmul(h, w_i)
        y += self.fc2.bias.view([1, -1])

        #
        return y.view(x_size)


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
        # routing
        x_size = x.size()
        x = x.view(
            [-1, self.d_model]
        )
        prob = self.router(x)
        topk = torch.topk(
            prob, k=self.n_blocks // 2,
            dim=-1, sorted=False
        )
        indices = topk.indices

        # blocking
        weight_gate = self.gate.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        lora_gate_r = self.gate.lora.right.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_side = self.side.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        lora_side_r = self.side.lora.right.weight.view(
            [self.n_blocks, self.block_size, -1]
        )
        weight_down = torch.permute(
            self.down.weight.view(
                [-1, self.n_blocks, self.block_size]
            ), dims=[1, 2, 0]
        )
        lora_down_l = self.down.lora.left.weight.view(
            [self.n_blocks, self.block_size, -1]
        )

        # applying
        y = torch.zeros_like(x)
        for i in range(self.n_blocks):
            cmp = torch.eq(indices, i)
            mask = torch.sum(
                cmp, dim=-1, dtype=torch.bool
            )
            coeff = 2.0 * prob[mask, i].unsqueeze(-1)
            # fc1
            x_i = x[mask]
            gate_i = weight_gate[i]
            side_i = weight_side[i]
            h_gate = coeff * torch.matmul(x_i, gate_i.T) + torch.matmul(
                torch.matmul(x_i, self.gate.lora.left.weight), lora_gate_r[i].T
            )
            h_side = coeff * torch.matmul(x_i, side_i.T) + torch.matmul(
                torch.matmul(x_i, self.side.lora.left.weight), lora_side_r[i].T
            )
            h = self.activation(h_gate) * h_side
            # fc2
            down_i = weight_down[i]
            y[mask] += coeff * torch.matmul(h, down_i) + torch.matmul(
                torch.matmul(h, lora_down_l[i]), self.down.lora.right.weight.T
            )

        #
        return y.view(x_size)
