import time
import torch
import random
from torch import nn
from torch import autograd
from tqdm import tqdm


class BLKMM(autograd.Function):
    @staticmethod
    def forward(ctx,
                config: dict,
                indptr: torch.Tensor,
                indices: torch.Tensor,
                weight: torch.Tensor,
                x: torch.Tensor):
        n_rows = len(indptr) - 1
        block_size = config['block_size']
        x_width = x.size(1) // block_size
        x_height = x.size(0) // block_size
        #
        xs = [
            x[
                h * block_size:(h + 1) * block_size,
                w * block_size:(w + 1) * block_size
            ]
            for h in range(x_height) for w in range(x_width)
        ]
        #
        output = []
        for row in range(n_rows):
            y_row = torch.zeros(
                [block_size, x_width * block_size],
                device=x.device
            )
            for i in range(indptr[row],
                           indptr[row + 1]):
                col = indices[i]
                wb = weight[
                    row * block_size:(row + 1) * block_size,
                    col * block_size:(col + 1) * block_size
                ]
                for j in range(x_width):
                    y_row[
                        :, j * block_size:(j + 1) * block_size
                    ] += torch.matmul(wb, xs[col * x_width + j])
            #
            output.append(y_row)
        output = torch.cat(output, dim=0)
        #
        ctx.xs = xs
        ctx.config = config
        ctx.save_for_backward(indptr, indices, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        xs: list = ctx.xs
        config: dict = ctx.config
        indptr: torch.Tensor = ctx.saved_tensors[0]
        indices: torch.Tensor = ctx.saved_tensors[1]
        weight: torch.Tensor = ctx.saved_tensors[2]
        #
        n_rows = len(indptr) - 1
        w_width = config['w_width']
        block_size = config['block_size']
        x_width = grad_output.size(1) // block_size
        output_width = grad_output.size(1) // block_size
        output_height = grad_output.size(0) // block_size
        grad_outputs = [
            grad_output[
                h * block_size:(h + 1) * block_size,
                w * block_size:(w + 1) * block_size
            ]
            for h in range(output_height) for w in range(output_width)
        ]
        grad_x = torch.zeros(
            [w_width * block_size, grad_output.size(-1)]
        )
        grad_w = torch.zeros_like(weight)
        #
        for row in range(n_rows):
            for i in range(indptr[row],
                           indptr[row + 1]):
                col = indices[i]
                for j in range(x_width):
                    # grad x
                    temp = torch.matmul(
                        weight[
                            row * block_size:(row + 1) * block_size,
                            col * block_size:(col + 1) * block_size
                        ].T, grad_outputs[row * x_width + j]
                    )
                    grad_x[
                        col * block_size:(col + 1) * block_size,
                        j * block_size:(j + 1) * block_size
                    ] += temp
                    # grad w
                    temp = torch.matmul(
                        xs[col * x_width + j],
                        grad_outputs[row * x_width + j],
                    ).T
                    grad_w[
                        row * block_size:(row + 1) * block_size,
                        col * block_size:(col + 1) * block_size
                    ] += temp
        #
        return None, None, None, grad_w, grad_x


def blkmm_fn(config, indptr, indices, weight, x):
    return BLKMM.apply(
        config, indptr, indices, weight, x
    )


def test_blkmm():
    block_size = 64
    in_features = 4 * block_size
    out_features = 16 * block_size
    batch_size = 16 * block_size

    # fc
    x = torch.randn(
        [batch_size, in_features],
        device='cuda', requires_grad=True
    )
    fc = nn.Linear(
        in_features, out_features, bias=False
    ).to('cuda')
    assert fc.weight.dim() == 2
    w_width = fc.weight.size(1) // block_size
    w_height = fc.weight.size(0) // block_size

    # blocks
    block_prob = torch.rand(
        size=[w_height, w_width]
    )
    block_mask = torch.where(
        block_prob < 0.25, True, False
    ).to(device='cuda')

    # sparse
    indptr, indices = [], []
    for row in range(w_height):
        indptr.append(len(indices))
        for col in torch.nonzero(block_mask[row]):
            indices.append(col.item())
    indptr.append(len(indices))
    indptr = torch.LongTensor(indptr).to(device='cuda')
    indices = torch.LongTensor(indices).to(device='cuda')

    # mask
    mask = block_mask.repeat_interleave(
        repeats=block_size, dim=-1
    )
    mask = mask.repeat_interleave(
        repeats=block_size, dim=0
    )
    mask = mask.to(
        dtype=torch.float, device='cuda'
    )

    # builtin
    time.sleep(2.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        y_1 = torch.matmul(
            torch.multiply(fc.weight, mask), x.T
        )
        # torch.sum(y_1).backward()
    torch.cuda.synchronize()
    print('timing 0:', time.time() - before)

    # kernel
    time.sleep(5.0)
    torch.cuda.synchronize()
    before = time.time()
    for _ in range(20):
        y_2: torch.Tensor = blkmm_fn(
            config={
                'w_width': w_width, 'w_height': w_height,
                'block_size': block_size, 'out_features': out_features
            },
            indptr=indptr, indices=indices, weight=fc.weight, x=x.T
        )
        # torch.sum(y_2).backward()
    torch.cuda.synchronize()
    print('timing 1:', time.time() - before)


def main():
    test_blkmm()


if __name__ == '__main__':
    main()
