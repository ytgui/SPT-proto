#include "common.h"

#define TSZ 8
#define BSZ 64
#define MAX_LEN 2048

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

    // windows
    for (index_t offset_k = 0; offset_k < d_head; offset_k += TSZ) {
        // cache lhs
        __shared__ scalar_t cache_lhs[BSZ][TSZ];
        cache_lhs[ty][tx] = lhs[gy * d_head + (offset_k + tx)];

        // cache rhs
        __shared__ scalar_t cache_rhs[MAX_LEN][TSZ];
        for (index_t offset_x = 0; offset_x < seq_length; offset_x += BSZ) {
            index_t gx = offset_x + ty;
            cache_rhs[offset_x + ty][tx] = rhs[gx * d_head + (offset_k + tx)];
        }
        __syncthreads();

        // contract
        for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += TSZ) {
            index_t gx = indices[i + tx];

            // product
            scalar_t reduced = 0.0;
            for (index_t k = 0; k < TSZ; k += 1) {
                reduced += cache_lhs[ty][k] * cache_rhs[gx][k];
            }
            output[i + tx] += reduced;
        }
        __syncthreads();
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
    TORCH_CHECK((d_head % TSZ) == 0);
    index_t seq_length = query.size(0);
    TORCH_CHECK((seq_length % BSZ) == 0);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // dispatch
    dim3 threads(TSZ, BSZ);
    dim3 blocks(1, seq_length / BSZ);
    sddmm_forward_cuda_kernel<float, float4><<<blocks, threads>>>(
        seq_length, d_head, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), query.data_ptr<float>(),
        key.data_ptr<float>(), output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}