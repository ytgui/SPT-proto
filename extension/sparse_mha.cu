#include "common.h"

template <typename scalar_t>
__global__ void mha_forward_cuda_kernel(
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *q,
    const scalar_t *k, const scalar_t *v, scalar_t *attention, scalar_t *output
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
        attention[sp_offset + cursor * n_heads + h] = reduced;
    }

    // softmax
    scalar_t denominator = 1.0 / cumulated;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        attention[sp_offset + cursor * n_heads + h] *= denominator;
    }
    __syncthreads();

    // apply
    for (index_t i = 0; i < d_head; i += 1) {
        scalar_t reduced = 0.0;
        for (index_t cursor = indptr[row]; cursor < indptr[row + 1];
             cursor += 1) {
            index_t col = indices[sp_offset + cursor * n_heads + h];
            reduced += attention[sp_offset + cursor * n_heads + h] *
                       v[qk_offset + col * n_heads * d_head + h * d_head + i];
        }
        output[qk_offset + row * n_heads * d_head + h * d_head + i] = reduced;
    }
}

template <typename scalar_t>
__global__ void mha_backward_cuda_kernel(
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *q,
    const scalar_t *k, const scalar_t *v, scalar_t *attention,
    const scalar_t *output, const scalar_t *grad_output,
    scalar_t *grad_attention, scalar_t *grad_q, scalar_t *grad_k,
    scalar_t *grad_v
) {
    // index
    index_t i = threadIdx.x;
    index_t h = threadIdx.y;
    index_t row = blockIdx.x;
    index_t n = blockIdx.y;
    index_t sp_offset = n * n_nonzeros * n_heads;
    index_t qk_offset = n * seq_length * n_heads * d_head;

    // apply gradient
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];
        scalar_t grad_row =
            grad_output[qk_offset + row * n_heads * d_head + h * d_head + i];
        atomicAdd(
            &grad_attention[sp_offset + cursor * n_heads + h],
            grad_row * v[qk_offset + col * n_heads * d_head + h * d_head + i]
        );
        atomicAdd(
            &grad_v[qk_offset + col * n_heads * d_head + h * d_head + i],
            grad_row * attention[sp_offset + cursor * n_heads + h]
        );
    }
    __syncthreads();

    // softmax gradient
    scalar_t cache = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        cache += attention[sp_offset + cursor * n_heads + h] *
                 grad_attention[sp_offset + cursor * n_heads + h];
    }

    // dot gradient
    scalar_t reduced = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];
        scalar_t grad_softmax =
            attention[sp_offset + cursor * n_heads + h] *
            (grad_attention[sp_offset + cursor * n_heads + h] - cache);
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

std::vector<torch::Tensor> sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value
) {
    CHECK_DIM(key, 4);
    CHECK_DIM(value, 4);
    CHECK_DIM(query, 4);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 3);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.sizes() == value.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());
    TORCH_CHECK(query.scalar_type() == value.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(2);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_nonzeros = indices.size(1);
    TORCH_CHECK(indices.size(-1) == n_heads);
    TORCH_CHECK(indices.size(0) == batch_size);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto attention = torch::zeros_like(indices, query.options());
    auto output = torch::zeros_like(value, query.options());

    // dispatch
    dim3 threads(n_heads);
    dim3 blocks(seq_length, batch_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "mha_forward_cuda_kernel", ([&] {
            mha_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                n_nonzeros, seq_length, n_heads, d_head,
                indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
                query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(), attention.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return {attention, output};
}

std::vector<torch::Tensor> sparse_mha_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, const torch::Tensor &attention,
    const torch::Tensor &output, const torch::Tensor &grad_output
) {
    CHECK_DIM(key, 4);
    CHECK_DIM(value, 4);
    CHECK_DIM(query, 4);
    CHECK_DIM(output, 4);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 3);
    CHECK_DIM(grad_output, 4);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(value.sizes() == grad_output.sizes());
    TORCH_CHECK(indices.sizes() == attention.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());
    TORCH_CHECK(query.scalar_type() == value.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t n_heads = query.size(2);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_nonzeros = indices.size(1);
    TORCH_CHECK(indices.size(-1) == n_heads);
    TORCH_CHECK(indices.size(0) == batch_size);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto grad_attention = torch::zeros_like(attention);
    auto grad_value = torch::zeros_like(value);
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    // dispatch
    dim3 threads(d_head, n_heads);
    dim3 blocks(seq_length, batch_size);
    // error: no instance of overloaded function "atomicAdd" matches the argument list (half)
    AT_DISPATCH_FLOATING_TYPES(
        grad_query.scalar_type(), "mha_backward_cuda_kernel", ([&] {
            mha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                n_nonzeros, seq_length, n_heads, d_head,
                indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
                query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(), attention.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
                grad_attention.data_ptr<scalar_t>(),
                grad_query.data_ptr<scalar_t>(), grad_key.data_ptr<scalar_t>(),
                grad_value.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return {grad_query, grad_key, grad_value};
}
