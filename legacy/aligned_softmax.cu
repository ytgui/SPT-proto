#include "common.h"

#define TSZ 4
#define BSZ 16

// clang-format off
template <typename scalar_t, typename vector_t>
__global__ void softmax_forward_kernel(
    index_t seq_length, index_t nonzeros, const index_t *indptr,
    const index_t *indices, const scalar_t *values, scalar_t *output) {
    // index
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gz = blockIdx.z * blockDim.z;

    // cumulate
    scalar_t cumulated = 0.0;
    for (index_t i = indptr[gz * (seq_length + 1) + gy];
         i < indptr[gz * (seq_length + 1) + gy + 1]; i += TSZ) {
        vector_t cache = __ldg(
            (const vector_t *)&values[gz * nonzeros + i]
        );
        cumulated += expf(cache.x) + expf(cache.y);
        cumulated += expf(cache.z) + expf(cache.w);
    }
    cumulated = fmax(1e-9, cumulated);

    // softmax
    scalar_t scale = 1.0 / cumulated;
    for (index_t i = indptr[gz * (seq_length + 1) + gy];
         i < indptr[gz * (seq_length + 1) + gy + 1]; i += TSZ) {
        vector_t cache = __ldg(
            (const vector_t *)&values[gz * nonzeros + i]
        );
        cache.x = scale * expf(cache.x);
        cache.y = scale * expf(cache.y);
        cache.z = scale * expf(cache.z);
        cache.w = scale * expf(cache.w);
        __stcs(
            (vector_t *)&output[gz * nonzeros + i], cache
        );
    }
}

template <typename scalar_t, typename vector_t>
__global__ void softmax_backward_kernel(
    index_t seq_length, index_t nonzeros, const index_t *indptr,
    const index_t *indices, const scalar_t *values, const scalar_t *output,
    const scalar_t *grad_output, scalar_t *grad_values) {
    // index
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gz = blockIdx.z * blockDim.z;

    // cumulate
    scalar_t cumulated = 0.0;
    for (index_t i = indptr[gz * (seq_length + 1) + gy];
         i < indptr[gz * (seq_length + 1) + gy + 1]; i += TSZ) {
        vector_t cache_output = __ldg(
            (const vector_t *)&output[gz * nonzeros + i]
        );
        vector_t cache_grad_output = __ldg(
            (const vector_t *)&grad_output[gz * nonzeros + i]
        );
        cumulated += cache_output.x * cache_grad_output.x;
        cumulated += cache_output.y * cache_grad_output.y;
        cumulated += cache_output.z * cache_grad_output.z;
        cumulated += cache_output.w * cache_grad_output.w;
    }
    cumulated = fmax(1e-9, cumulated);

    // gradient
    for (index_t i = indptr[gz * (seq_length + 1) + gy];
         i < indptr[gz * (seq_length + 1) + gy + 1]; i += TSZ) {
        vector_t cache_output = __ldg(
            (const vector_t *)&output[gz * nonzeros + i]
        );
        vector_t cache_grad_output = __ldg(
            (const vector_t *)&grad_output[gz * nonzeros + i]
        );
        cache_output.x = cache_output.x * (cache_grad_output.x - cumulated);
        cache_output.y = cache_output.y * (cache_grad_output.y - cumulated);
        cache_output.z = cache_output.z * (cache_grad_output.z - cumulated);
        cache_output.w = cache_output.w * (cache_grad_output.w - cumulated);
        __stcs(
            (vector_t *)&grad_values[gz * nonzeros + i], cache_output
        );
    }
}
// clang-format on

torch::Tensor softmax_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values
) {
    CHECK_DIM(indptr, 2);
    CHECK_DIM(indices, 2);
    CHECK_DIM(values, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(indptr.size(0) == indices.size(0));
    TORCH_CHECK(indices.sizes() == values.sizes());

    // sizes
    index_t nonzeros = indices.size(-1);
    index_t batch_size = indptr.size(0);
    index_t seq_length = indptr.size(1) - 1;
    TORCH_CHECK((seq_length % BSZ) == 0);
    auto output = torch::empty_like(values);

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, seq_length / BSZ, batch_size);
    softmax_forward_kernel<float, float4><<<blocks, threads>>>(
        seq_length, nonzeros, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), values.data_ptr<float>(),
        output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}

torch::Tensor softmax_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values, const torch::Tensor &output,
    const torch::Tensor &grad_output
) {
    CHECK_DIM(indptr, 2);
    CHECK_DIM(indices, 2);
    CHECK_DIM(values, 2);
    CHECK_DIM(output, 2);
    CHECK_DIM(grad_output, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(indptr.size(0) == indices.size(0));
    TORCH_CHECK(grad_output.sizes() == output.sizes());
    TORCH_CHECK(indices.sizes() == values.sizes());
    TORCH_CHECK(values.sizes() == output.sizes());

    // sizes
    index_t nonzeros = indices.size(-1);
    index_t batch_size = indptr.size(0);
    index_t seq_length = indptr.size(1) - 1;
    TORCH_CHECK((seq_length % BSZ) == 0);
    auto grad_values = torch::empty_like(values);

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, seq_length / BSZ, batch_size);
    softmax_backward_kernel<float, float4><<<blocks, threads>>>(
        seq_length, nonzeros, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), values.data_ptr<float>(),
        output.data_ptr<float>(), grad_output.data_ptr<float>(),
        grad_values.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return grad_values;
}
