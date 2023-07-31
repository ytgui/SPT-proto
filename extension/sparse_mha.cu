// clang-format off
#include <iostream>
#include <torch/extension.h>
// clang-format on

#define BLOCK_SIZE 16

#define CHECK_DIM(x, d)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")                 \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

#define CHECK_TYPE(x, t) \
    TORCH_CHECK(x.scalar_type() == t, #x " must be type of " #t)

using index_t = int64_t;

template <typename scalar_t>
__global__ void sparse_mha_forward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *q, const scalar_t *k,
    scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gy = blockIdx.y * blockDim.y + ty;
    for (index_t gi = indptr[gy]; gi < indptr[gy + 1]; gi += 1) {
        index_t gx = indices[gi];

        // window
        scalar_t reduced = 0.0;
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            reduced += q[gy * d_head + i] * k[gx * d_head + i];
        }

        // store
        output[gi] = reduced;
    }
}

torch::Tensor sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_DIM(key, 2);
    CHECK_DIM(query, 2);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes(), "query.size() != key.size()");
    TORCH_CHECK(
        query.scalar_type() == key.scalar_type(), "scalar_type is different"
    );

    // sizes
    index_t d_head = query.size(-1);
    index_t seq_length = query.size(0);
    TORCH_CHECK(seq_length % BLOCK_SIZE == 0, "seq_length is not aligned");
    TORCH_CHECK(
        seq_length + 1 == indptr.size(0), "indptr doesn't match seq_length"
    );
    auto output = torch::zeros({indices.size(0)}, query.options());

    // dispatch
    index_t dt = BLOCK_SIZE;
    index_t db = seq_length / BLOCK_SIZE;
    dim3 threads(dt, dt), blocks(1, db);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "sparse_mha_forward_cuda", ([&] {
        sparse_mha_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            seq_length, d_head, indptr.data_ptr<index_t>(),
            indices.data_ptr<index_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>()
        );
        TORCH_CHECK(cudaGetLastError() == cudaSuccess);
      })
    );

    //
    return output;
}
