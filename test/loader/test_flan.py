import os
from naive_gpt import loaders


def test_flanmini():
    batch_size = 16
    seq_length = 512

    #
    dm = loaders.FlanMiniDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=seq_length + 1,
        batch_size=batch_size,
        tokenizer='opt',
        num_workers=1
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.size(0) == batch_size
    assert batch.size(-1) == seq_length + 2

    # decode
    for sample in batch:
        text = dm.tokenizer.decode(sample)
        encoded = dm.tokenizer.encode(text)
        union = set(sample.tolist()).union(encoded)
        inter = set(sample.tolist()).intersection(encoded)
        assert abs(len(inter) - len(union)) < 20

    #
    print('[PASS] test_flanmini()')


def main():
    test_flanmini()


if __name__ == '__main__':
    main()
