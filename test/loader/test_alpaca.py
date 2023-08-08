import os
from naive_gpt import loaders


def test_alpaca():
    batch_size = 16
    seq_length = 1024
    alpaca_prefix = 'Below is an instruction that describes a task'

    #
    dm = loaders.AlpacaDataModule(
        root=os.getenv('HOME') +
        '/Public/Datasets/text/',
        seq_length=seq_length + 1,
        batch_size=batch_size,
        num_workers=1
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.size(0) == batch_size
    assert batch.size(-1) == seq_length + 1

    # decode
    tokenizer = dm.tokenizer
    for sample in batch:
        text = tokenizer.decode(sample)
        assert alpaca_prefix in text
        assert '<pad>' in text
        assert '</s>' in text

    #
    print('[PASS] test_alpaca()')


def main():
    test_alpaca()


if __name__ == '__main__':
    main()
