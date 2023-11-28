#include "common.h"

#define TILE_SIZE 4
#define BLOCK_SIZE 16
#define WORKER_SIZE 4

using vector_t = int4;

// clang-format off
template <unsigned N_SPACES, unsigned NNZ_SIZE>
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
    __shared__ uint16_t cache_lhs[BLOCK_SIZE][N_SPACES];
    for (index_t k = tx; k < N_SPACES; k += WORKER_SIZE) {
        cache_lhs[ty][k] = left[
            gz * seq_length * N_SPACES + gy * N_SPACES + k
        ];
    }

    // output
    #define N_SLOTS 4
    index_t cursors[N_SLOTS] = {tx, tx, tx, tx};
    __shared__ uint16_t indices[BLOCK_SIZE][N_SLOTS][NNZ_SIZE];

    // window
    for (index_t offset_x = 0; offset_x < seq_length; offset_x += BLOCK_SIZE) {
        if (offset_x > (gy - ty)) {
            break;
        }
        // load rhs
        __shared__ uint16_t cache_rhs[BLOCK_SIZE][N_SPACES];
        for (index_t k = tx; k < N_SPACES; k += WORKER_SIZE) {
            cache_rhs[ty][k] = right[
                gz * seq_length * N_SPACES + (offset_x + ty) * N_SPACES + k
            ];
        }
        __syncthreads();

        // lookup
        for (index_t local_x = tx; local_x < BLOCK_SIZE; local_x += WORKER_SIZE) {
            // tril
            if ((offset_x + local_x) > gy) {
                break;
            }
            // count
            index_t count = 0;
            for (index_t k = 0; k < N_SPACES; k += 1) {
                count += (
                    cache_lhs[ty][k] == cache_rhs[local_x][k]
                );
            }
            index_t slot = min(
                N_SLOTS - 1, count / (N_SPACES / N_SLOTS)
            );
            index_t cursor = cursors[slot];
            indices[ty][slot][cursor] = offset_x + local_x;
            cursors[slot] = min(cursor + WORKER_SIZE, NNZ_SIZE - tx - 1);
        }
        __syncthreads();
    }

    // store
    index_t slot = N_SLOTS - 1, cursor = tx;
    index_t offset_b = gz * seq_length * nonzeros;
    for (index_t local_x = tx; local_x < min(gy + 1, nonzeros); local_x += WORKER_SIZE) {
        while (slot >= 0 && cursor >= cursors[slot]) {
            slot = slot - 1; cursor = tx;
        }
        if (slot < 0) {
            break;
        }
        output[offset_b + gy * nonzeros + local_x] = indices[ty][slot][cursor];
        cursor = cursor + WORKER_SIZE;
    }
}
// clang-format on

torch::Tensor lookup_forward_cuda(
    const torch::Tensor &config, const torch::Tensor &query,
    const torch::Tensor &key
) {
    CHECK_DIM(key, 3);
    CHECK_DIM(query, 3);
    CHECK_TYPE(key, torch::kInt32);
    CHECK_TYPE(query, torch::kInt32);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t sparsity = config.size(0);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t n_subspaces = query.size(-1);
    TORCH_CHECK(seq_length % BLOCK_SIZE == 0);
    TORCH_CHECK(seq_length % sparsity == 0);
    index_t nonzeros = seq_length / sparsity;
    TORCH_CHECK(nonzeros % BLOCK_SIZE == 0);
    auto output = torch::zeros(
        {batch_size, seq_length, nonzeros}, query.options()
    );

    // dispatch
    dim3 blocks(1, seq_length / BLOCK_SIZE, batch_size);
    if (n_subspaces == 8 && nonzeros == 32) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<8, 32><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 8 && nonzeros == 64) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<8, 64><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 8 && nonzeros == 128) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<8, 128><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 10 && nonzeros == 32) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<10, 32><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 10 && nonzeros == 64) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<10, 64><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 10 && nonzeros == 128) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<10, 128><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 16 && nonzeros == 32) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<16, 32><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 16 && nonzeros == 64) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<16, 64><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 16 && nonzeros == 128) {
        dim3 threads(WORKER_SIZE, BLOCK_SIZE);
        lookup_forward_kernel<16, 128><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            key.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else {
        TORCH_CHECK(false && "n_subspaces not supported");
    }
    CUDA_CHECH(cudaGetLastError());

    //
    return output;
}