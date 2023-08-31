#include "common.h"

#define BK 16
#define BM 64
#define BN BM

template <typename scalar_t>
__global__ void sddmm_forward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *lhs, const scalar_t *rhs,
    scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache
    scalar_t cache_lhs[BK];
    __shared__ scalar_t cache_rhs[BN][BK];

    // k-loop
    for (index_t offset_k = 0; offset_k < d_head; offset_k += BK) {
        // sparse
        index_t cursor = indptr[gy];
        index_t cursor_limit = indptr[gy + 1];

        // n-loop
        for (index_t offset_n = 0; offset_n < seq_length; offset_n += BN) {
            // load
            for (index_t k = 0; k < BK; k += 1) {
                cache_lhs[k] = lhs[
                    gy * d_head + (offset_k + k)
                ];
                cache_rhs[ty][k] = rhs[
                    (offset_n + ty) * d_head + (offset_k + k)
                ];
            }
            __syncthreads();

            // contract
            while (cursor < cursor_limit) {
                index_t col = indices[cursor];
                if (col >= (offset_n + BN)) {
                    break;
                }

                // product
                scalar_t reduced = 0.0;
                for (index_t k = 0; k < BK; k += 1) {
                    reduced += cache_lhs[k] * cache_rhs[col % BN][k];
                }

                // store
                output[cursor] += reduced;
                cursor += 1;
            }
            __syncthreads();
        }
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