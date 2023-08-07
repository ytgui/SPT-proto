import os
import random
from naive_gpt import loaders


def test_mmlu():
    batch_size = 1
    seq_length = 256

    #
    dm = loaders.MMLUModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        n_shots=random.randrange(10),
        seq_length=seq_length + 1,
        batch_size=batch_size,
        num_workers=1
    )
    loader = dm.val_dataloader()
    sample = next(iter(loader))
    assert len(sample) == 1
    text = sample[0]

    # decode
    tokenizer = dm.tokenizer
    src, target = text[:-1], text[-1]
    target = tokenizer.decode(target)
    src = tokenizer.decode(src)
    target = target.lstrip(' ')
    assert target in [
        'A', 'B', 'C', 'D'
    ]

    #
    print('----- Q -----\n', src)
    print('----- A -----\n', target)
    print('[PASS] test_mmlu()')


def main():
    test_mmlu()


if __name__ == '__main__':
    main()
