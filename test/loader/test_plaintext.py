import os
import random
import string
from torch.utils import data
from naive_gpt import loaders


def test_santize():
    # case 1
    for i in range(10):
        src = ''.join(
            ['\n' for _ in range(i)]
        )
        output = loaders.Sanitize()(src)
        assert output == ''

    # case 2
    src = """
      [] Advanced Micro Devices, Inc., commonly abbreviated as AMD, is an American multinational
    semiconductor company based in Santa  Clara , California (), that develops computer processors
    and related technologies for business and consumer markets.\n\n\n\n\n
    AMD's main products include microprocessors, motherboard chipsets, embedded processors, graphics
    processors, and FPGAs for servers, workstations, personal computers, and embedded system applications.\n
    """
    target = 'Advanced Micro Devices, Inc., commonly abbreviated as AMD, is an American multinational ' + \
        'semiconductor company based in Santa Clara, California, that develops computer processors and ' + \
        'related technologies for business and consumer markets.\n\n' + \
        "AMD's main products include microprocessors, motherboard chipsets, embedded processors, graphics " + \
        'processors, and FPGAs for servers, workstations, personal computers, and embedded system applications.'
    output = loaders.Sanitize()(src)
    assert output == target

    #
    print('[PASS] test_santize()')


def test_padding():
    seq_length = random.randint(16, 256)
    pad_fn = loaders.ClampPadding(
        seq_length=seq_length, pad_value=0xff
    )

    #
    x = [
        random.randint(0, 100)
        for _ in range(seq_length // 2)
    ]
    y = pad_fn(x)

    # check
    assert len(y) == seq_length
    assert y[-1] == 0xff

    #
    print('[PASS] test_padding()')


def main():
    test_santize()
    test_padding()


if __name__ == '__main__':
    main()
