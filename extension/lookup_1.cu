#include "common.h"

#define TK 4
#define BM 16

using vector_t = int4;

// clang-format off
template <unsigned BK>
__global__ void lookup_forward_kernel(
    index_t batch_size, index_t seq_length, index_t nonzeros,
    const index_t *left, const index_t *right, index_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache lhs
    __shared__ index_t cache_lhs[BM][BK];
    for (uint16_t k = tx * TK; k < BK; k += BK) {
        *(vector_t *)&cache_lhs[ty][k] = __ldg(
            (const vector_t *)&left[
                gz * seq_length * BK + gy * BK + k
            ]
        );
    }

    // window
    for (uint16_t offset_x = 0; offset_x < seq_length; offset_x += BM) {
        // cache rhs
        __shared__ index_t cache_rhs[BM][BK];
        for (uint16_t k = tx * TK; k < BK; k += BK) {
            *(vector_t *)&cache_rhs[ty][k] = __ldg(
                (const vector_t *)&right[
                    gz * seq_length * BK + (offset_x + ty) * BK + k
                ]
            );
        }
        __syncthreads();

        // cache output
        uint16_t cursors[4] = {};
        __shared__ uint8_t cache_indices[BM][4][BM];

        // lookup
        for (uint16_t local_x = tx; local_x < BM; local_x += blockDim.x) {
            uint16_t count = 0;
            for (uint16_t k = 0; k < BK; k += 1) {
                count += (
                    cache_lhs[ty][k] == cache_rhs[local_x][k]
                );
            }
            count = min(7, count) / 2;
            uint16_t slot = 32 - __clz(count);
            cache_indices[ty][slot][local_x] = count;
        }
        __syncthreads();

        // store
        for (uint16_t local_x = tx * TK; local_x < BM; local_x += blockDim.x * TK) {
            index_t buffer[TK];
            for (index_t t = 0; t < TK; t += 1) {
                buffer[t] = cache_indices[ty][t][local_x];
            }
            __stcs(
                (vector_t *)&output[
                    gz * seq_length * nonzeros + gy * nonzeros + local_x
                ], *(const vector_t *)buffer
            );
        }
    }
}
// clang-format on

torch::Tensor lookup_forward_cuda(
    const torch::Tensor &config, const torch::Tensor &query,
    const torch::Tensor &store
) {
    CHECK_DIM(query, 3);
    CHECK_DIM(store, 3);
    CHECK_TYPE(query, torch::kInt32);
    CHECK_TYPE(store, torch::kInt32);
    TORCH_CHECK(query.sizes() == store.sizes());
    TORCH_CHECK(query.scalar_type() == store.scalar_type());

    // sizes
    index_t sparsity = config.size(0);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_subspaces = query.size(-1);
    TORCH_CHECK(seq_length % BM == 0);
    TORCH_CHECK(seq_length % sparsity == 0);
    index_t nonzeros = seq_length / sparsity;
    TORCH_CHECK(nonzeros % BM == 0);
    auto output = torch::empty(
        {batch_size, seq_length, nonzeros}, query.options()
    );

    // dispatch
    dim3 blocks(1, seq_length / BM, batch_size);
    if (n_subspaces == 8) {
        const auto BK = 8;
        dim3 threads(BK / TK, BM);
        lookup_forward_kernel<BK><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            store.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else {
        TORCH_CHECK(false && "n_subspaces not supported");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}