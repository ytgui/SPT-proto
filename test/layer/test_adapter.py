import torch
import random
from torch import nn
from naive_gpt import layers, utils


def test_upgrader():
    n_features = random.randint(1, 1024)
    batch_size = random.randint(1, 64)

    # input
    x_1 = torch.randn(
        [batch_size, n_features],
        requires_grad=True
    )
    x_2 = x_1.detach().clone()
    x_2.requires_grad = True

    # model
    for model_1, model_2 in [
        # depth 1
        [
            nn.Sequential(
                nn.Linear(n_features, 10),
                nn.Tanh(),
                nn.Softmax(dim=-1)
            ),
            nn.Sequential(
                nn.Linear(n_features, 10),
                nn.Softmax(dim=-1)
            )
        ],
        # depth 2
        [
            nn.Sequential(
                nn.Sequential(
                    nn.Linear(n_features, 10),
                    nn.Tanh()
                ),
                nn.Softmax(dim=-1)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Linear(n_features, 10)
                ),
                nn.Softmax(dim=-1)
            )
        ]
    ]:
        model_2.load_state_dict(
            model_1.state_dict()
        )

        # replace
        class ModuleHandler:
            def default(self,
                        name: str,
                        child: nn.Module):
                pass

            def onLinear(self,
                         name: str,
                         child: nn.Module):
                new_child = nn.Sequential(
                    child, nn.Tanh()
                )
                return new_child

        upgrader = utils.ModuleUpgrader(
            handler=ModuleHandler()
        )
        model_2 = upgrader.visit(root=model_2)

        # compare
        y_1 = model_1(x_1)
        y_2 = model_2(x_2)
        torch.sum(y_1).backward()
        torch.sum(y_2).backward()
        assert torch.allclose(
            y_1, y_2, atol=1e-5, rtol=1e-3
        )
        assert torch.allclose(
            x_1.grad, x_2.grad, atol=1e-5, rtol=1e-3
        )

    #
    print('[PASS] test_upgrader()')


def test_upgrade_opt():
    d_lora = random.randint(1, 16)
    block_size = 16 * random.randint(1, 4)
    d_model = block_size * random.randint(1, 2)
    d_feedforward = block_size * random.choice([8, 16])
    batch_size = random.randint(1, 64)

    # x
    x = torch.randn(
        [batch_size, d_model]
    )

    # model 1
    model_0 = layers.Feedforward(
        d_model=d_model,
        d_feedforward=d_feedforward,
        activation=nn.SiLU(),
        p_dropout=0.0
    )
    upgrader = utils.ModuleUpgrader(
        handler=utils.SparseLoRAHandler(
            d_lora=d_lora, stage='lora'
        )
    )
    model_0 = upgrader.visit(model_0)
    model_1 = layers.RoutedFFN.from_pretrained(
        block_size=block_size, source=model_0
    )

    # model 2
    model_2 = layers.LoRARoutedFFN.from_pretrained(
        d_lora, block_size=block_size, source=model_0
    )
    model_2.router.load_state_dict(model_1.router.state_dict())

    # lora zero output
    y_1, y_2 = model_1(x), model_2(x)
    assert torch.allclose(y_1, y_2, atol=1e-3)

    # freeze parameters
    parameters = list(
        filter(
            lambda v: v.requires_grad,
            model_2.parameters()
        )
    )
    assert len(parameters) == 6

    #
    print('[PASS] test_upgrade_opt()')


def test_upgrade_llama():
    d_lora = random.randint(1, 16)
    block_size = 16 * random.randint(1, 4)
    d_model = block_size * random.randint(1, 2)
    d_feedforward = block_size * random.choice([8, 16])
    batch_size = random.randint(1, 64)

    # x
    x = torch.randn(
        [batch_size, d_model]
    )

    # model 1
    model_0 = layers.LLaMaFeedforward(
        d_model=d_model,
        d_feedforward=d_feedforward,
        activation=nn.SiLU()
    )
    upgrader = utils.ModuleUpgrader(
        handler=utils.SparseLoRAHandler(
            d_lora=d_lora, stage='lora'
        )
    )
    model_0 = upgrader.visit(model_0)
    model_1 = layers.RoutedLLaMaFFN.from_pretrained(
        block_size=block_size, source=model_0
    )

    # model 2
    model_2 = layers.LoRARoutedLLaMaFFN.from_pretrained(
        d_lora, block_size=block_size, source=model_0
    )
    model_2.router.load_state_dict(model_1.router.state_dict())

    # lora zero output
    y_1, y_2 = model_1(x), model_2(x)
    assert torch.allclose(y_1, y_2, atol=1e-5)

    # freeze parameters
    parameters = list(
        filter(
            lambda v: v.requires_grad,
            model_2.parameters()
        )
    )
    assert len(parameters) == 8

    #
    print('[PASS] test_upgrade_llama()')


def main():
    test_upgrader()
    test_upgrade_opt()
    test_upgrade_llama()


if __name__ == '__main__':
    main()
