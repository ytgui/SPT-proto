#include "common.h"

#define TILE_SIZE 4
#define BLOCK_SIZE 32
#define WORKER_SIZE 2

using vector_t = int4;

// clang-format off
template <unsigned N_SPACES>
__global__ void lookup_forward_kernel(
    index_t batch_size, index_t seq_length, index_t nonzeros,
    const index_t *left, const index_t *right, index_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // load lhs
    __shared__ index_t cache_lhs[BLOCK_SIZE][N_SPACES];
    for (index_t k = tx * TILE_SIZE; k < N_SPACES; k += WORKER_SIZE * TILE_SIZE) {
        *(vector_t *)&cache_lhs[ty][k] = __ldg(
            (const vector_t *)&left[
                gz * seq_length * N_SPACES + gy * N_SPACES + k
            ]
        );
    }

    // window
    index_t offset_output = 0;
    for (index_t offset_x = 0; offset_x < seq_length; offset_x += BLOCK_SIZE) {
        // load rhs
        __shared__ index_t cache_rhs[BLOCK_SIZE][N_SPACES];
        for (index_t k = tx * TILE_SIZE; k < N_SPACES; k += WORKER_SIZE * TILE_SIZE) {
            *(vector_t *)&cache_rhs[ty][k] = __ldg(
                (const vector_t *)&right[
                    gz * seq_length * N_SPACES + (offset_x + ty) * N_SPACES + k
                ]
            );
        }
        __syncthreads();

        // lookup
        index_t cursors[4] = {tx, tx, tx, tx};
        __shared__ uint16_t indices[BLOCK_SIZE][4][BLOCK_SIZE];
        for (index_t local_x = tx; local_x < BLOCK_SIZE; local_x += WORKER_SIZE) {
            index_t count = 0;
            for (index_t k = 0; k < N_SPACES; k += 1) {
                count += (
                    cache_lhs[ty][k] == cache_rhs[local_x][k]
                );
            }
            count = min(3, count / 2);
            index_t slot = 32 - __clz(count);
            index_t cursor = cursors[slot];
            indices[ty][slot][cursor] = offset_x + local_x;
            cursors[slot] = min(cursor + WORKER_SIZE, BLOCK_SIZE - tx - 1);
        }
        __syncthreads();

        // store
        index_t slot = 4 - 1, cursor = tx;
        index_t n_times = seq_length / BLOCK_SIZE;
        index_t offset_b = gz * seq_length * nonzeros;
        for (index_t local_x = tx; local_x < nonzeros / n_times; local_x += WORKER_SIZE) {
            while (cursor >= cursors[slot]) {
                slot = slot - 1; cursor = tx;
            }
            index_t indice = indices[ty][slot][cursor];
            output[offset_b + gy * nonzeros + offset_output + local_x] = indice;
            cursor = cursor + WORKER_SIZE;
        }
        offset_output += nonzeros / n_times;
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
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
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