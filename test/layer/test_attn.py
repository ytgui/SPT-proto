import torch
import random
from torch import nn
from naive_gpt import layers


def test_vanilla_attn():
    n_heads = random.choice(
        [1, 2, 4, 8]
    )
    batch_size = random.randint(1, 16)
    seq_length = random.randint(1, 64)
    d_model = 8 * random.randint(1, 64)

    # set precision
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)

    # init model
    mha_1 = nn.MultiheadAttention(
        d_model, num_heads=n_heads,
        dropout=0.0, bias=True,
        batch_first=True
    )
    mha_2 = layers.MultiheadAttention(
        d_model, n_heads=n_heads,
        attention_fn=layers.VanillaAttention(
            d_head=d_model // n_heads,
            p_dropout=0.0
        )
    )

    # init parameters
    with torch.no_grad():
        # qkv bias
        mha_1.in_proj_bias.set_(
            torch.cat([
                mha_2.linear_q.bias,
                mha_2.linear_k.bias,
                mha_2.linear_v.bias
            ], dim=0)
        )
        # qkv weight
        mha_1.in_proj_weight.set_(
            torch.cat([
                mha_2.linear_q.weight,
                mha_2.linear_k.weight,
                mha_2.linear_v.weight
            ], dim=0)
        )
        # output
        mha_1.out_proj.bias.set_(mha_2.linear_o.bias)
        mha_1.out_proj.weight.set_(mha_2.linear_o.weight)

    #
    q = torch.randn(
        [batch_size, seq_length, d_model],
    )
    k = torch.randn(
        [batch_size, seq_length, d_model]
    )
    v = torch.randn(
        [batch_size, seq_length, d_model]
    )
    y_1, y_2 = mha_1(q, k, v)[0], mha_2(q, k, v)
    assert torch.allclose(y_1, y_2, atol=1e-8, rtol=1e-5)

    # pytest will not reset the process
    torch.set_default_dtype(old_dtype)

    #
    print('[PASS] test_vanilla_attn()')


def test_rotary_attn():
    n_heads = random.choice(
        [1, 2, 4, 8]
    )
    batch_size = random.randint(1, 16)
    seq_length = random.randint(1, 64)
    d_head = 8 * random.randint(1, 64)
    d_model = n_heads * d_head

    # init model
    mha_fn = layers.MultiheadAttention(
        d_model=d_model, n_heads=n_heads,
        attention_fn=layers.RotaryAttention(
            d_head=d_head, p_dropout=0.0
        )
    )
    x = torch.ones(
        [batch_size, seq_length, d_model],
    )
    mha_fn(x, x, x)

    #
    print('[PASS] test_rotary_attn()')


def main():
    test_vanilla_attn()
    test_rotary_attn()


if __name__ == '__main__':
    main()
