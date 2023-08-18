#include "common.h"

template <typename scalar_t>
__global__ void cdist_forward_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gx = blockIdx.x * blockDim.x + tx;

    // window
    scalar_t reduced = 0.0;
    for (index_t offset = 0; offset < d_code; offset += BLOCK_SIZE) {
        // cache
        __shared__ scalar_t cache_q[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t cache_t[BLOCK_SIZE][BLOCK_SIZE];

        // store
        cache_q[ty][tx] = query[gy * d_code + (offset + tx)];
        cache_t[ty][tx] = table[gx * d_code + (offset + ty)];
        __syncthreads();

        // product
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            reduced += fabsf(cache_q[ty][i] - cache_t[i][tx]);
        }
        __syncthreads();
    }

    // store
    output[gy * n_codewords + gx] = reduced;
}

torch::Tensor cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
) {
    CHECK_DIM(query, 2);
    CHECK_DIM(table, 2);
    TORCH_CHECK(query.size(-1) == table.size(-1));
    TORCH_CHECK(query.scalar_type() == table.scalar_type());

    // sizes
    index_t d_code = query.size(-1);
    index_t n_queries = query.size(0);
    index_t n_codewords = table.size(0);
    TORCH_CHECK(d_code % BLOCK_SIZE == 0);
    TORCH_CHECK(n_queries % BLOCK_SIZE == 0);
    TORCH_CHECK(n_codewords % BLOCK_SIZE == 0);
    auto output = torch::zeros({n_queries, n_codewords}, query.options());

    // dispatch
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n_codewords / BLOCK_SIZE, n_queries / BLOCK_SIZE);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "cdist_forward_kernel", ([&] {
            cdist_forward_kernel<scalar_t><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<scalar_t>(),
                table.data_ptr<scalar_t>(), output.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return output;
}