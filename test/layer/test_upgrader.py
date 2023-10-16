import torch
import random
from torch import nn
from naive_gpt import utils


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


def main():
    test_upgrader()


if __name__ == '__main__':
    main()
