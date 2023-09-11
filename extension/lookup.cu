#include "common.h"

#define TSZ 4
#define BSZ 64
using vector_t = int4;

// clang-format off
template <unsigned NSP, unsigned NNZ>
__global__ void lookup_forward_kernel(
    index_t batch_size, index_t seq_length, index_t nonzeros,
    const index_t *query, const index_t *store, index_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache
    index_t cache_query[NSP];
    for (index_t k = 0; k < NSP; k += TSZ) {
        *(vector_t *)&cache_query[k] = __ldg(
            (const vector_t *)&query[
                gz * seq_length * NSP + gy * NSP + k
            ]
        );
    }

    // result
    index_t cache_sizes[NSP / 2] = {};
    index_t cache_indices[NSP / 2][NNZ];

    // window
    for (index_t offset_x = 0; offset_x < seq_length; offset_x += BSZ) {
        // cache
        __shared__ index_t cache_store[BSZ][NSP];
        for (index_t k = 0; k < NSP; k += TSZ) {
            *(vector_t *)&cache_store[ty][k] = __ldg(
                (const vector_t *)&store[
                    gz * seq_length * NSP + (offset_x + ty) * NSP + k
                ]
            );
        }
        __syncthreads();

        // product
        for (index_t tx = 0; tx < BSZ; tx += 1) {
            index_t count = 0;
            for (index_t k = 0; k < NSP; k += 1) {
                count += (cache_query[k] == cache_store[tx][k]);   
            }
            index_t slot = min(count / 2, NSP / 2 - 1);
            index_t cursor = min(cache_sizes[slot], NNZ - 1);
            cache_indices[slot][cursor] = offset_x + tx;
            cache_sizes[slot] += 1;
        }
        __syncthreads();
    }

    // store
    index_t slot = NSP / 2 - 1, cursor = 0;
    for (index_t gx = 0; gx < nonzeros; gx += TSZ) {
        index_t buffer[TSZ];
        for (index_t t = 0; t < TSZ; t += 1) {
            buffer[t] = cache_indices[slot][cursor];
            // move slot and cursor
            index_t limit = cache_sizes[slot];
            slot -= (cursor + 1) / limit;
            cursor = (cursor + 1) % limit;
        }
        __stcs(
            (vector_t *)&output[
                gz * seq_length * nonzeros + gy * nonzeros + gx
            ], *(const vector_t *)buffer
        );
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
    TORCH_CHECK(seq_length % BSZ == 0);
    TORCH_CHECK(n_subspaces % TSZ == 0);
    TORCH_CHECK(seq_length % sparsity == 0);
    index_t nonzeros = seq_length / sparsity;
    TORCH_CHECK(nonzeros % BSZ == 0);
    auto output = torch::empty(
        {batch_size, seq_length, nonzeros}, query.options()
    );

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, seq_length / BSZ, batch_size);
    if (n_subspaces == 8) {
        if (seq_length / sparsity <= 64) {
            lookup_forward_kernel<8, 64><<<blocks, threads>>>(
                batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
                store.data_ptr<index_t>(), output.data_ptr<index_t>()
            );
        } else if (seq_length / sparsity <= 256) {
            lookup_forward_kernel<8, 256><<<blocks, threads>>>(
                batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
                store.data_ptr<index_t>(), output.data_ptr<index_t>()
            );
        } else {
            TORCH_CHECK(false && "seq_length / sparsity not supported");
        }
    } else {
        TORCH_CHECK(false && "n_subspaces not supported");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}