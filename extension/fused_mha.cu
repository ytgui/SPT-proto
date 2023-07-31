// clang-format off
#include <iostream>
#include <torch/extension.h>
// clang-format on

#define BLOCK_SIZE 16

#define CHECK_INPUT(x, d)                                                 \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")                 \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

using index_t = int32_t;

template <typename scalar_t>
__global__ void sparse_mha_forward_cuda(
    index_t seq_length, index_t d_head, const scalar_t *q, const scalar_t *k,
    scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gx = blockIdx.x * blockDim.x + tx;

    // window
    scalar_t reduced = 0.0;
    for (index_t offset = 0; offset < d_head; offset += BLOCK_SIZE) {
        // cache
        __shared__ scalar_t cache_q[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t cache_k[BLOCK_SIZE][BLOCK_SIZE];

        // store
        // a[gy, tx], b[ty, gx], bT[gx, ty]
        cache_q[ty][tx] = q[gy * d_head + (offset + tx)];
        cache_k[ty][tx] = k[gx * d_head + (offset + ty)];
        __syncthreads();

        // product
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            reduced += cache_q[ty][i] * cache_k[i][tx];
        }
    }

    // store
    output[gy * seq_length + gx] = reduced;
}

torch::Tensor sparse_mha_forward(
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_INPUT(query, 2);
    CHECK_INPUT(key, 2);
    TORCH_CHECK(query.sizes() == key.sizes(), "query.size() != key.size()");

    // sizes
    index_t d_head = query.size(-1);
    index_t seq_length = query.size(0);
    TORCH_CHECK(seq_length % BLOCK_SIZE == 0, "seq_length is not aligned");
    auto output = torch::zeros({seq_length, seq_length}, query.options());

    // dispatch
    index_t dt = BLOCK_SIZE;
    index_t db = seq_length / BLOCK_SIZE;
    dim3 threads(dt, dt), blocks(db, db);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        output.type(), "sparse_mha_forward_cuda", ([&] {
            sparse_mha_forward_cuda<scalar_t><<<blocks, threads>>>(
                seq_length, d_head, query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>()
            );
        })
    );

    //
    return output;
}
