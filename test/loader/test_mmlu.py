import os
import random
from naive_gpt import loaders


def test_mmlu():
    batch_size = 1
    seq_length = random.choice(
        [128, 256, 512]
    )

    #
    dm = loaders.MMLUDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        n_shots=random.randrange(10),
        max_length=seq_length + 1,
        batch_size=batch_size,
        tokenizer='opt',
        num_workers=1
    )
    loader = dm.val_dataloader()

    #
    for i, sample in enumerate(loader):
        if i == 16:
            break
        assert len(sample) == 1
        text = sample[0]
        assert len(text) == seq_length + 2

        # decode
        tokenizer = dm.tokenizer
        pos, src = text[0], text[1:-1]
        target = text[pos]
        target = tokenizer.decode(target)
        src = tokenizer.decode(src)
        target = target.lstrip(' ')
        assert target in [
            'A', 'B', 'C', 'D'
        ]

        #
        print('----- Q -----\n', src)
        print('----- A -----\n', target)

    #
    print('[PASS] test_mmlu()')


def main():
    test_mmlu()


if __name__ == '__main__':
    main()
