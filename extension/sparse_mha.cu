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
    index_t seq_length, index_t n_heads, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *q, const scalar_t *k,
    scalar_t *output
) {
    // index
    index_t h = threadIdx.x;
    index_t row = blockIdx.x;

    // contract
    for (index_t cursor = indptr[row * n_heads + h];
         cursor < indptr[(row + 1) * n_heads + h]; cursor += 1) {
        index_t col = indices[cursor * n_heads + h];

        // product
        scalar_t reduced = 0.0;
        for (index_t i = 0; i < d_head; i += 1) {
            reduced += q[row * n_heads * d_head + h * d_head + i] *
                       k[col * n_heads * d_head + h * d_head + i];
        }

        // store
        output[cursor * n_heads + h] = reduced;
    }
}

template <typename scalar_t>
__global__ void sparse_mha_backward_cuda_kernel(
    index_t seq_length, index_t n_heads, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *q, const scalar_t *k,
    const scalar_t *grad_output, scalar_t *grad_q, scalar_t *grad_k
) {
    // index
    index_t i = threadIdx.x;
    index_t h = threadIdx.y;
    index_t row = blockIdx.x;

    // gradient
    scalar_t reduced = 0.0;
    for (index_t cursor = indptr[row * n_heads + h];
         cursor < indptr[(row + 1) * n_heads + h]; cursor += 1) {
        index_t col = indices[cursor * n_heads + h];
        atomicAdd(
            &grad_k[col * n_heads * d_head + h * d_head + i],
            grad_output[cursor * n_heads + h] *
                q[row * n_heads * d_head + h * d_head + i]
        );
        reduced += grad_output[cursor * n_heads + h] *
                   k[col * n_heads * d_head + h * d_head + i];
    }
    grad_q[row * n_heads * d_head + h * d_head + i] = reduced;
}

torch::Tensor sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_DIM(key, 3);
    CHECK_DIM(query, 3);
    CHECK_DIM(indptr, 2);
    CHECK_DIM(indices, 2);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(1);
    index_t seq_length = query.size(0);
    index_t n_nonzeros = indices.size(0);
    TORCH_CHECK(n_heads == indptr.size(-1));
    TORCH_CHECK(seq_length + 1 == indptr.size(0));
    TORCH_CHECK(indptr.size(-1) == indices.size(-1));
    auto output = torch::zeros({n_nonzeros, n_heads}, query.options());

    // dispatch
    dim3 threads(n_heads);
    dim3 blocks(seq_length);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        output.scalar_type(), "sparse_mha_forward_cuda", ([&] {
            sparse_mha_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                seq_length, n_heads, d_head, indptr.data_ptr<index_t>(),
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
    CHECK_DIM(key, 3);
    CHECK_DIM(query, 3);
    CHECK_DIM(indptr, 2);
    CHECK_DIM(indices, 2);
    CHECK_DIM(grad_output, 2);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(indices.sizes() == grad_output.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(1);
    index_t seq_length = query.size(0);
    index_t n_nonzeros = indices.size(0);
    TORCH_CHECK(n_heads == indptr.size(-1));
    TORCH_CHECK(seq_length + 1 == indptr.size(0));
    TORCH_CHECK(indptr.size(-1) == indices.size(-1));
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    // dispatch
    dim3 threads(d_head, n_heads);
    dim3 blocks(seq_length);
    AT_DISPATCH_FLOATING_TYPES(
        grad_query.scalar_type(), "sparse_mha_backward_cuda", ([&] {
            sparse_mha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                seq_length, n_heads, d_head, indptr.data_ptr<index_t>(),
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
