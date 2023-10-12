import torch
import random
from torch import nn
from naive_gpt import models, tuning
from naive_torch.models import ModuleUpgrader


def test_upgrade():
    ckpt = torch.load(
        f='.data/opt-125m.ckpt'
    )
    d_lora = random.choice([16, 32])
    p_dropout = random.random()

    #
    config = ckpt['config']
    model = models.OPTModel(**config)
    model.load_state_dict(ckpt['state_dict'])

    # insert LoRA
    upgrader_1 = ModuleUpgrader(
        handler=tuning.LoRAUpgrader(
            lora_r=d_lora,
            lora_dropout=p_dropout
        )
    )
    model = upgrader_1.visit(model)

    # insert quantizer
    upgrader_2 = ModuleUpgrader(
        handler=tuning.QuantizedUpgrader(
            d_codeword=4, n_codewords=64
        )
    )
    model = upgrader_2.visit(model)

    # check
    n_trains = 0
    n_parameters = 0
    for p in model.parameters():
        p: nn.Parameter
        if p.requires_grad:
            n_trains += p.numel()
        n_parameters += p.numel()
    print('n_trains:', n_trains)
    print('n_parameters:', n_parameters)
    print('ratio:', n_trains / n_parameters)
    assert n_trains / n_parameters < 0.05

    #
    print('[PASS] test_upgrade()')


def main():
    test_upgrade()


if __name__ == '__main__':
    main()
