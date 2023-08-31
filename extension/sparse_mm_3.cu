#include "common.h"

#define BK 16
#define BM 16

template <typename scalar_t, typename vector_t>
__global__ void sddmm_forward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *lhs, const scalar_t *rhs,
    scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache
    vector_t cache_lhs[BK / 4];
    for (index_t k = 0; k < BK; k += 4) {
        cache_lhs[k / 4] = __ldg(
            (const vector_t *)&lhs[gy * d_head + k]
        );
    }

    // contract
    for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += 1) {
        index_t gx = indices[i];

        // product
        scalar_t reduced = 0.0;
        for (index_t k = 0; k < BK; k += 4) {
            const vector_t cache_rhs = __ldg(
                (const vector_t *)&rhs[gx * d_head + k]
            );
            reduced += cache_lhs[k / 4].x * cache_rhs.x;
            reduced += cache_lhs[k / 4].y * cache_rhs.y;
            reduced += cache_lhs[k / 4].z * cache_rhs.z;
            reduced += cache_lhs[k / 4].w * cache_rhs.w;
        }

        // store
        output[i] = reduced;
    }
}

torch::Tensor sddmm_forward_cuda(
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
    TORCH_CHECK((d_head % BK) == 0);
    index_t seq_length = query.size(0);
    TORCH_CHECK((seq_length % BM) == 0);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // dispatch
    dim3 threads(1, BM);
    dim3 blocks(1, seq_length / BM);
    sddmm_forward_cuda_kernel<float, float4><<<blocks, threads>>>(
        seq_length, d_head, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), query.data_ptr<float>(),
        key.data_ptr<float>(), output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}