import os
import random
from naive_gpt import loaders


def test_glue():
    seq_length = random.choice(
        [128, 256, 512]
    )
    batch_size = random.randint(1, 16)

    # Single sentence: CoLA, SST-2
    # Similarity Paraphrase: QQP, STS-B, MRPC
    # Statement Inference: QNLI MNLI RTE
    for glue_subset in ['cola', 'qqp', 'qnli']:
        dm = loaders.GLUEDataModule(
            root=os.getenv('HOME') +
            '/Public/Datasets/text',
            subset=glue_subset,
            seq_length=seq_length,
            batch_size=batch_size,
            tokenizer='bert-base-uncased',
            num_workers=1
        )
        loader = dm.val_dataloader()

        #
        sample = next(iter(loader))
        assert len(sample) == 3
        assert sample[0].size() == sample[1].size()

    #
    print('[PASS] test_glue()')


def main():
    test_glue()


if __name__ == '__main__':
    main()
