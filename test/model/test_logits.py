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

    # load from ckpt
    ckpt = torch.load(
        '.data/opt-125m.ckpt'
    )
    config = ckpt['config']
    opt_2 = models.OPTModel(**config)
    opt_2.load_state_dict(ckpt['state_dict'])
    opt_2.eval()

    # check output
    batch_size = 1
    seq_length = 256
    for _ in tqdm(range(16)):
        x = torch.randint(
            high=config['vocab_size'],
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
