import torch


def main():
    # load
    raw_ckpt = torch.load(
        '.data/opt-125m.ckpt', map_location='cpu'
    )
    tuned_ckpt = torch.load(
        'lightning_logs/version_9/checkpoints/LM-epoch=1-ppl=575.0-accuracy=0.244.ckpt',
        map_location='cpu'
    )
    raw_state_dict = raw_ckpt['state_dict']
    tuned_state_dict = tuned_ckpt['state_dict']

    # filter
    total_size = 0
    filtered_dict = {}
    for name in tuned_state_dict:
        param: torch.Tensor
        param = tuned_state_dict[name]
        name = name.removeprefix('model.')
        if name in raw_state_dict:
            if torch.allclose(
                raw_state_dict[name],
                param, atol=1e-3
            ):
                continue
        print('[SAVE]', name)
        filtered_dict[name] = param
        total_size += param.numel()

    # check
    assert total_size < (50 * 1024 * 1024)
    torch.save({'state_dict': filtered_dict}, '.data/spt.ckpt')


if __name__ == '__main__':
    main()
