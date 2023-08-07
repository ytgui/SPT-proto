import torch
import transformers as T
from naive_gpt import models
from tqdm import tqdm


def test_logits():
    # huggingface
    OptModel = T.OPTForCausalLM
    opt_1 = OptModel.from_pretrained(
        'facebook/opt-125m'
    ).eval()
    config = opt_1.config

    # load from state dict
    opt_2 = models.OPTModel(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        p_dropout=config.dropout,
        vocab_size=config.vocab_size,
        d_embedding=config.word_embed_proj_dim,
        d_feedforward=config.ffn_dim,
        max_length=config.max_position_embeddings
    )
    opt_2.load_state_dict(
        torch.load('.data/opt-125m.ckpt')
    )
    opt_2.eval()

    # check output
    batch_size = 1
    seq_length = 256
    for _ in tqdm(range(16)):
        x = torch.randint(
            high=config.vocab_size,
            size=[batch_size, seq_length]
        )
        y_1, y_2 = opt_1(x)['logits'], opt_2(x)
        assert torch.allclose(
            y_1, y_2, atol=1e-5, rtol=1e-3
        )

    #
    print('[PASS] test_logits()')


def main():
    test_logits()


if __name__ == '__main__':
    main()
