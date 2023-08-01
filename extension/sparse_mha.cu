// clang-format off
#include <vector>
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
    index_t gy = blockIdx.y * blockDim.y + ty;
    for (index_t gi = indptr[gy]; gi < indptr[gy + 1]; gi += 1) {
        index_t gx = indices[gi];

        // product
        scalar_t reduced = 0.0;
        for (index_t i = 0; i < d_head; i += 1) {
            reduced += q[gy * d_head + i] * k[gx * d_head + i];
        }

        // store
        output[gi] = reduced;
    }
}

template <typename scalar_t>
__global__ void sparse_mha_backward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *q, const scalar_t *k,
    const scalar_t *grad_output, scalar_t *grad_q, scalar_t *grad_k
) {
    // index
    index_t tx = threadIdx.x;
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // gradient
    scalar_t reduced = 0.0;
    for (index_t gi = indptr[gy]; gi < indptr[gy + 1]; gi += 1) {
        index_t gx = indices[gi];
        atomicAdd(
            &grad_k[gx * d_head + tx], grad_output[gi] * q[gy * d_head + tx]
        );
        reduced += grad_output[gi] * k[gx * d_head + tx];
    }
    grad_q[gy * d_head + tx] = reduced;
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
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t seq_length = query.size(0);
    // TORCH_CHECK(seq_length % BLOCK_SIZE == 0);
    TORCH_CHECK(seq_length + 1 == indptr.size(0));
    auto output = torch::zeros({indices.size(0)}, query.options());

    // dispatch
    index_t dt = 1;
    index_t db = seq_length / dt;
    dim3 threads(dt), blocks(1, db);
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

std::vector<torch::Tensor> sparse_mha_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &grad_output
) {
    CHECK_DIM(key, 2);
    CHECK_DIM(query, 2);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_DIM(grad_output, 1);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(indices.sizes() == grad_output.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t seq_length = query.size(0);
    // TORCH_CHECK(seq_length % BLOCK_SIZE == 0);
    TORCH_CHECK(seq_length + 1 == indptr.size(0));
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    // dispatch
    dim3 threads(d_head);
    dim3 blocks(1, seq_length);
    AT_DISPATCH_FLOATING_TYPES(
        grad_query.scalar_type(), "sparse_mha_backward_cuda", ([&] {
            sparse_mha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                seq_length, d_head, indptr.data_ptr<index_t>(),
                indices.data_ptr<index_t>(), query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
                grad_query.data_ptr<scalar_t>(), grad_key.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return {grad_query, grad_key};
}
