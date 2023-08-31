#include "common.h"

#define TSZ 4
#define BSZ 64

template <typename scalar_t, typename vector_t>
__global__ void sddmm_forward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *lhs, const scalar_t *rhs,
    scalar_t *output
) {
    // index
    index_t tx = threadIdx.x;
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache
    __shared__ vector_t cache_lhs[BSZ / TSZ];
    cache_lhs[tx] = __ldg(
        (const vector_t *)&lhs[gy * d_head + tx * TSZ]
    );
    __syncthreads();

    // contract
    for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += BSZ) {
        scalar_t reduced[TSZ] = {0.0};
        for (index_t t = 0; t < TSZ; t += 1) {
            index_t gx = indices[i + tx * TSZ + t];

            // product
            for (index_t k = 0; k < BSZ; k += 4) {
                const vector_t cache_rhs = __ldg(
                    (const vector_t *)&rhs[gx * d_head + k]
                );
                reduced[t] += cache_lhs[k / 4].x * cache_rhs.x;
                reduced[t] += cache_lhs[k / 4].y * cache_rhs.y;
                reduced[t] += cache_lhs[k / 4].z * cache_rhs.z;
                reduced[t] += cache_lhs[k / 4].w * cache_rhs.w;
            }
        }
        __stcg(
            (vector_t *)&output[i + tx * TSZ], *(const vector_t *)reduced
        );
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
    TORCH_CHECK((d_head % BSZ) == 0);
    index_t seq_length = query.size(0);
    TORCH_CHECK((seq_length % BSZ) == 0);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // dispatch
    dim3 threads(BSZ / TSZ);
    dim3 blocks(1, seq_length);
    sddmm_forward_cuda_kernel<float, float4><<<blocks, threads>>>(
        seq_length, d_head, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), query.data_ptr<float>(),
        key.data_ptr<float>(), output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}