#include "common.h"

#define TILE_SIZE 4
#define BLOCK_SIZE 32

using vector_t = int4;

// clang-format off
template <unsigned N_SPACES>
__global__ void lookup_forward_kernel(
    index_t batch_size, index_t seq_length, index_t nonzeros,
    const index_t *left, const index_t *right, index_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache lhs
    index_t cache_lhs[N_SPACES];
    for (index_t k = 0; k < N_SPACES; k += TILE_SIZE) {
        *(vector_t *)&cache_lhs[k] = __ldg(
            (const vector_t *)&left[
                gz * seq_length * N_SPACES + gy * N_SPACES + k
            ]
        );
    }

    // window
    for (index_t offset_x = 0; offset_x < seq_length; offset_x += BLOCK_SIZE) {
        // cache rhs
        __shared__ index_t cache_rhs[BLOCK_SIZE][N_SPACES];
        for (index_t k = 0; k < N_SPACES; k += TILE_SIZE) {
            *(vector_t *)&cache_rhs[ty][k] = __ldg(
                (const vector_t *)&right[
                    gz * seq_length * N_SPACES + (offset_x + ty) * N_SPACES + k
                ]
            );
        }
        __syncthreads();

        // lookup
        index_t cursors[3] = {};
        uint8_t indices[3][BLOCK_SIZE];
        for (index_t local_x = 0; local_x < BLOCK_SIZE; local_x += 1) {
            index_t count = 0;
            for (index_t k = 0; k < N_SPACES; k += 1) {
                count += (
                    cache_lhs[k] == cache_rhs[local_x][k]
                );
            }
            count = min(7, count) / 2;
            index_t slot = 32 - __clz(count);
            index_t cursor = cursors[slot];
            indices[slot][cursor] = local_x;
            cursors[slot] = min(cursor + 1, BLOCK_SIZE - 1);
        }

        // store
        index_t slot = 4 - 1, cursor = 0;
        index_t n_times = seq_length / BLOCK_SIZE;
        for (index_t local_x = 0; local_x < nonzeros / n_times; local_x += 4) {
            index_t buffer[TILE_SIZE];
            for (index_t t = 0; t < TILE_SIZE; t += 1) {
                buffer[t] = indices[slot][cursor];
                bool cond = (cursor == cursors[slot] - 1);
                cursor = cond ? 0 : (cursor + 1);
                slot = cond ? (slot - 1) : slot;
            }
            __stcs(
                (vector_t *)&output[
                    gz * seq_length * nonzeros + gy * nonzeros + local_x
                ], *(const vector_t *)buffer
            );
        }
        __syncthreads();
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
    TORCH_CHECK(seq_length % BLOCK_SIZE == 0);
    TORCH_CHECK(seq_length % sparsity == 0);
    index_t nonzeros = seq_length / sparsity;
    TORCH_CHECK(nonzeros % BLOCK_SIZE == 0);
    auto output = torch::empty(
        {batch_size, seq_length, nonzeros}, query.options()
    );

    // dispatch
    dim3 blocks(1, seq_length / BLOCK_SIZE, batch_size);
    if (n_subspaces == 8) {
        dim3 threads(1, BLOCK_SIZE);
        lookup_forward_kernel<8><<<blocks, threads>>>(
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