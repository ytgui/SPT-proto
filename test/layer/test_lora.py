import torch
import random
from torch import nn
from naive_gpt import layers


def test_lora_linear():
    d_model = random.randint(1, 16)
    in_features = random.randint(1, 256)
    out_features = random.randint(1, 256)
    batch_size = random.randint(1, 64)

    # init
    x = torch.randn(
        [batch_size, in_features]
    )
    model_1 = nn.Linear(
        in_features=in_features,
        out_features=out_features
    )
    model_2 = layers.LoRALinear.from_pretrained(
        d_model, p_dropout=0.0, source=model_1
    )

    # lora zero output
    y_1 = model_1(x)
    y_2 = model_2(x)
    assert torch.allclose(
        y_1, y_2, atol=1e-5, rtol=1e-3
    )

    # freeze parameters
    parameters = list(
        filter(
            lambda v: v.requires_grad,
            model_2.parameters()
        )
    )
    assert len(parameters) == 2

    #
    print('[PASS] test_lora_linear()')


def test_lora_embedding():
    d_model = random.randint(1, 16)
    num_embeddings = random.randint(1, 1024)
    embedding_dim = random.randint(1, 256)
    batch_size = random.randint(1, 64)

    # init
    x = torch.randint(
        high=num_embeddings,
        size=[batch_size]
    )
    model_1 = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    )
    model_2 = layers.LoRAEmbedding(
        d_model=d_model,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        lora_dropout=0.0
    )
    model_2 = layers.LoRAEmbedding.from_pretrained(
        d_model, p_dropout=0.0, source=model_1
    )

    # lora zero output
    y_1 = model_1(x)
    y_2 = model_2(x)
    assert torch.allclose(
        y_1, y_2, atol=1e-5, rtol=1e-3
    )

    # freeze parameters
    parameters = list(
        filter(
            lambda v: v.requires_grad,
            model_2.parameters()
        )
    )
    assert len(parameters) == 2

    #
    print('[PASS] test_lora_embedding()')


def test_lora_routed():
    d_model = random.randint(1, 16)
    in_features = random.randint(1, 256)
    block_size = 16 * random.randint(1, 4)
    out_features = block_size * random.randint(1, 4)
    batch_size = random.randint(1, 64)

    # init
    x = torch.randn(
        [batch_size, in_features]
    )
    model_1 = layers.RoutedFFN(
        block_size=block_size,
        in_features=in_features,
        out_features=out_features,
        actication=nn.SiLU()
    )
    model_2 = layers.LoRARoutedFFN.from_pretrained(
        d_model, p_dropout=0.0, source=model_1
    )

    # lora zero output
    y_1 = model_1(x)
    y_2 = model_2(x)
    assert torch.allclose(
        y_1, y_2, atol=1e-5, rtol=1e-3
    )

    # freeze parameters
    parameters = list(
        filter(
            lambda v: v.requires_grad,
            model_2.parameters()
        )
    )
    assert len(parameters) == 4

    #
    print('[PASS] test_lora_routed()')


def main():
    test_lora_linear()
    test_lora_embedding()
    test_lora_routed()


if __name__ == '__main__':
    main()
