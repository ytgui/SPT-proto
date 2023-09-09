#include "common.h"

#define TSZ 4
#define BSZ 16
#define MAX_NNZ 256
using vector_t = int4;

// clang-format off
template <unsigned NSZ>
__global__ void lookup_forward_kernel(
    index_t batch_size, index_t seq_length, index_t nonzeros,
    const index_t *query, const index_t *store, index_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // cache
    __shared__ index_t cache_query[BSZ][NSZ];
    for (index_t offset_k = 0; offset_k < NSZ; offset_k += TSZ) {
        *(vector_t *)&cache_query[ty, offset_k] = __ldg(
            (const vector_t *)&query[
                gz * seq_length * NSZ + gy * NSZ + offset_k
            ]
        );
    }

    // result
    index_t size_indices[NSZ / 2] = {};
    index_t cache_indices[NSZ / 2][MAX_NNZ];

    // window
    for (index_t offset_x = 0; offset_x < seq_length; offset_x += BSZ) {
        // cache
        __shared__ index_t cache_store[BSZ][NSZ];
        for (index_t offset_k = 0; offset_k < NSZ; offset_k += TSZ) {
            *(vector_t *)&cache_store[ty, offset_k] = __ldg(
                (const vector_t *)&store[
                    gz * seq_length * NSZ + (offset_x + ty) * NSZ + offset_k
                ]
            );
        }
        __syncthreads();

        // product
        for (index_t tx = 0; tx < BSZ; tx += 1) {
            index_t count = 0;
            for (index_t k = 0; k < NSZ; k += 1) {
                count += (cache_query[ty][k] == cache_store[tx][k]);   
            }
            index_t cursor = size_indices[count / 2];
            cache_indices[count / 2][cursor] = offset_x + tx;
            size_indices[count / 2] += 1;
        }
        __syncthreads();
    }

    // store
    index_t cursor_i = NSZ / 2 - 1, cursor_j = 0;
    for (index_t gx = 0; gx < nonzeros; gx += 1) {
        output[
            gz * seq_length * nonzeros + gy * nonzeros + gx
        ] = cache_indices[cursor_i][cursor_j];
        if (cursor_j >= size_indices[cursor_i]) {
            cursor_i -= 1; cursor_j = 0;
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
    CHECK_TYPE(query, torch::kFloat32);
    CHECK_TYPE(store, torch::kFloat32);
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
    TORCH_CHECK(nonzeros <= MAX_NNZ);
    auto output = torch::empty(
        {batch_size, seq_length, nonzeros}, query.options()
    );

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, seq_length / BSZ, batch_size);
    if (n_subspaces == 8) {
        lookup_forward_kernel<8><<<blocks, threads>>>(
            batch_size, seq_length, nonzeros, query.data_ptr<index_t>(),
            store.data_ptr<index_t>(), output.data_ptr<index_t>()
        );
    } else if (n_subspaces == 16) {
        lookup_forward_kernel<16><<<blocks, threads>>>(
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