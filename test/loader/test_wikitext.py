import os
from naive_gpt import loaders


def test_wikitext():
    batch_size = 16
    seq_length = 256

    #
    dm = loaders.WikitextDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=seq_length,
        batch_size=batch_size,
        tokenizer='opt',
        num_workers=1
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.size(0) == batch_size
    assert batch.size(-1) == seq_length

    # decode
    tokenizer = dm.tokenizer
    for sample in batch:
        assert len(sample) == seq_length
        text = tokenizer.decode(sample)
        encoded = tokenizer.encode(text)
        union = set(sample.tolist()).union(encoded)
        inter = set(sample.tolist()).intersection(encoded)
        assert abs(len(inter) - len(union)) < 20

    #
    print('[PASS] test_wikitext()')


def main():
    test_wikitext()


if __name__ == '__main__':
    main()
