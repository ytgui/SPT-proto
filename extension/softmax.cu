#include "common.h"

#define TSZ 4
#define BSZ 16

// clang-format off
template <typename scalar_t, typename vector_t>
__global__ void softmax_forward_kernel(
    index_t seq_length, const index_t *indptr, const index_t *indices,
    const scalar_t *values, scalar_t *output) {
    // index
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cumulate
    scalar_t cumulated = 0.0;
    for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += TSZ) {
        vector_t cache = __ldg(
            (const vector_t *)&values[i]
        );
        cumulated += expf(cache.x) + expf(cache.y);
        cumulated += expf(cache.z) + expf(cache.w);
    }

    // softmax
    scalar_t scale = 1.0 / cumulated;
    for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += TSZ) {
        vector_t cache = __ldg(
            (const vector_t *)&values[i]
        );
        cache.x = scale * expf(cache.x);
        cache.y = scale * expf(cache.y);
        cache.z = scale * expf(cache.z);
        cache.w = scale * expf(cache.w);
        __stcs((vector_t *)&output[i], cache);
    }
}
// clang-format on

torch::Tensor softmax_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values
) {
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(indices.sizes() == values.sizes());

    // sizes
    index_t seq_length = indptr.size(0) - 1;
    TORCH_CHECK((seq_length % BSZ) == 0);
    auto output = torch::zeros_like(values);

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, seq_length / BSZ);
    softmax_forward_kernel<float, float4><<<blocks, threads>>>(
        seq_length, indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
        values.data_ptr<float>(), output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}
