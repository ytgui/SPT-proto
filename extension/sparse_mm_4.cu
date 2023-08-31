#include "common.h"

#define BSZ 16

template <typename scalar_t>
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
    __shared__ scalar_t cache_lhs[BSZ];
    cache_lhs[tx] = lhs[gy * d_head + tx];
    __syncthreads();

    // contract
    for (index_t i = indptr[gy]; i < indptr[gy + 1]; i += BSZ) {
        index_t gx = indices[i + tx];

        // product
        scalar_t reduced = 0.0;
        for (index_t k = 0; k < d_head; k += 1) {
            reduced += cache_lhs[k] * rhs[gx * d_head + k];
        }
        output[i + tx] = reduced;
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
    dim3 threads(BSZ);
    dim3 blocks(1, seq_length);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "sddmm_forward_cuda_kernel", ([&] {
            sddmm_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
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