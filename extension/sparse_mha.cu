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
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *q,
    const scalar_t *k, scalar_t *output
) {
    // index
    index_t h = threadIdx.x;
    index_t row = blockIdx.x;
    index_t n = blockIdx.y;
    index_t sp_offset = n * n_nonzeros * n_heads;
    index_t qk_offset = n * seq_length * n_heads * d_head;

    // contract
    scalar_t cumulated = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];

        // product
        scalar_t reduced = 0.0;
        for (index_t i = 0; i < d_head; i += 1) {
            reduced += q[qk_offset + row * n_heads * d_head + h * d_head + i] *
                       k[qk_offset + col * n_heads * d_head + h * d_head + i];
        }
        reduced = expf(reduced);
        cumulated += reduced;

        // store
        output[sp_offset + cursor * n_heads + h] = reduced;
    }

    // softmax
    scalar_t denominator = 1.0 / cumulated;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        output[sp_offset + cursor * n_heads + h] *= denominator;
    }
}

template <typename scalar_t>
__global__ void sparse_mha_backward_cuda_kernel(
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *q,
    const scalar_t *k, const scalar_t *output, const scalar_t *grad_output,
    scalar_t *grad_q, scalar_t *grad_k
) {
    // index
    index_t i = threadIdx.x;
    index_t h = threadIdx.y;
    index_t row = blockIdx.x;
    index_t n = blockIdx.y;
    index_t sp_offset = n * n_nonzeros * n_heads;
    index_t qk_offset = n * seq_length * n_heads * d_head;

    // softmax gradient
    scalar_t cache = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        cache += output[sp_offset + cursor * n_heads + h] *
                 grad_output[sp_offset + cursor * n_heads + h];
    }

    // dot gradient
    scalar_t reduced = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];
        scalar_t grad_softmax =
            output[sp_offset + cursor * n_heads + h] *
            (grad_output[sp_offset + cursor * n_heads + h] - cache);
        atomicAdd(
            &grad_k[qk_offset + col * n_heads * d_head + h * d_head + i],
            grad_softmax *
                q[qk_offset + row * n_heads * d_head + h * d_head + i]
        );
        reduced += grad_softmax *
                   k[qk_offset + col * n_heads * d_head + h * d_head + i];
    }
    grad_q[qk_offset + row * n_heads * d_head + h * d_head + i] = reduced;
}

torch::Tensor sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_DIM(key, 4);
    CHECK_DIM(query, 4);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 3);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(2);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_nonzeros = indices.size(1);
    TORCH_CHECK(indices.size(-1) == n_heads);
    TORCH_CHECK(indices.size(0) == batch_size);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // dispatch
    dim3 threads(n_heads);
    dim3 blocks(seq_length, batch_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        output.scalar_type(), "sparse_mha_forward_cuda", ([&] {
            sparse_mha_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                n_nonzeros, seq_length, n_heads, d_head,
                indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
                query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>()
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
    const torch::Tensor &output, const torch::Tensor &grad_output
) {
    CHECK_DIM(key, 4);
    CHECK_DIM(query, 4);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 3);
    CHECK_DIM(output, 3);
    CHECK_DIM(grad_output, 3);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(indices.sizes() == output.sizes());
    TORCH_CHECK(indices.sizes() == grad_output.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(2);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_nonzeros = indices.size(1);
    TORCH_CHECK(indices.size(-1) == n_heads);
    TORCH_CHECK(indices.size(0) == batch_size);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    // dispatch
    dim3 threads(d_head, n_heads);
    dim3 blocks(seq_length, batch_size);
    AT_DISPATCH_FLOATING_TYPES(
        grad_query.scalar_type(), "sparse_mha_backward_cuda", ([&] {
            sparse_mha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                n_nonzeros, seq_length, n_heads, d_head,
                indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
                query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
                grad_query.data_ptr<scalar_t>(), grad_key.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return {grad_query, grad_key};
}
