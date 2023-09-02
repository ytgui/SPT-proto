#include "common.h"

#define BLOCK_SIZE 4

#define TSZ 4
#define BSZ 64

// clang-format off
template <typename scalar_t, typename vector_t>
__global__ void cdist_forward_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, scalar_t *output) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // window
    for (index_t offset_x = 0; offset_x < n_codewords; offset_x += BSZ) {
        scalar_t reduced[BSZ] = {};
        for (index_t offset_k = 0; offset_k < d_code; offset_k += TSZ) {
            // cache
            __shared__ vector_t cache_a[BSZ];
            __shared__ vector_t cache_b[BSZ];
            cache_a[ty] = __ldg(
                (const vector_t *)&query[
                    gz * n_queries * d_code + gy * d_code + offset_k
                ]
            );
            cache_b[ty] = __ldg(
                (const vector_t *)&table[
                    gz * n_codewords * d_code + (offset_x + ty) * d_code + offset_k
                ]
            );
            __syncthreads();

            // product
            for (index_t tx = 0; tx < BSZ; tx += 1) {
                reduced[tx] += fabsf(cache_a[ty].x - cache_b[tx].x);
                reduced[tx] += fabsf(cache_a[ty].y - cache_b[tx].y);
                reduced[tx] += fabsf(cache_a[ty].z - cache_b[tx].z);
                reduced[tx] += fabsf(cache_a[ty].w - cache_b[tx].w);
            }
            __syncthreads();
        }

        // store
        index_t offset_z = gz * n_queries * n_codewords;
        for (index_t tx = 0; tx < BSZ; tx += TSZ) {
            index_t gx = offset_x + tx;
            __stcs(
                (vector_t *)&output[offset_z + gy * n_codewords + gx],
                *(const vector_t *)&reduced[tx]
            );
        }
    }
}

template <typename scalar_t>
__global__ void cdist_backward_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, const scalar_t *grad_output,
    scalar_t *grad_query, scalar_t *grad_table) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gx = blockIdx.x * blockDim.x + tx;

    // load
    scalar_t value = grad_output[
        gz * n_queries * n_codewords + gy * n_codewords + gx
    ];

    // window
    for (index_t offset = 0; offset < d_code; offset += BLOCK_SIZE) {
        // cache
        __shared__ scalar_t cache_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t cache_b[BLOCK_SIZE][BLOCK_SIZE];

        // store
        cache_a[ty][tx] = query[
            gz * n_queries * d_code + gy * d_code + (offset + tx)
        ];
        cache_b[ty][tx] = table[
            gz * n_codewords * d_code + gx * d_code + (offset + ty)
        ];
        __syncthreads();

        // product
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            scalar_t v = (
                cache_a[ty][i] - cache_b[i][tx]
            ) > 0 ? value : -value;
            atomicAdd(&grad_query[
                gz * n_queries * d_code + gy * d_code + (offset + i)
            ], v);
            atomicAdd(&grad_table[
                gz * n_codewords * d_code + gx * d_code + (offset + i)
            ], -v);
        }
        __syncthreads();
    }
}
// clang-format on

torch::Tensor cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
) {
    CHECK_DIM(query, 3);
    CHECK_DIM(table, 3);
    TORCH_CHECK(query.size(0) == table.size(0));
    TORCH_CHECK(query.size(-1) == table.size(-1));
    TORCH_CHECK(query.scalar_type() == table.scalar_type());

    // sizes
    index_t d_code = table.size(-1);
    index_t n_queries = query.size(1);
    index_t n_codewords = table.size(1);
    index_t n_subspaces = table.size(0);
    TORCH_CHECK(d_code % TSZ == 0);
    TORCH_CHECK(n_queries % BSZ == 0);
    TORCH_CHECK(n_codewords % BSZ == 0);
    auto output = torch::zeros(
        {n_subspaces, n_queries, n_codewords}, query.options()
    );

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, n_queries / BSZ, n_subspaces);
    cdist_forward_kernel<float, float4><<<blocks, threads>>>(
        n_queries, n_codewords, d_code, query.data_ptr<float>(),
        table.data_ptr<float>(), output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}

std::vector<torch::Tensor> cdist_backward_cuda(
    const torch::Tensor &query, const torch::Tensor &table,
    const torch::Tensor &grad_output
) {
    CHECK_DIM(query, 3);
    CHECK_DIM(table, 3);
    CHECK_DIM(grad_output, 3);
    TORCH_CHECK(query.size(0) == table.size(0));
    TORCH_CHECK(query.size(-1) == table.size(-1));
    TORCH_CHECK(query.size(0) == grad_output.size(0));
    TORCH_CHECK(query.size(1) == grad_output.size(1));
    TORCH_CHECK(table.size(1) == grad_output.size(-1));
    TORCH_CHECK(query.scalar_type() == table.scalar_type());
    TORCH_CHECK(query.scalar_type() == grad_output.scalar_type());

    // sizes
    index_t d_code = table.size(-1);
    index_t n_queries = query.size(1);
    index_t n_codewords = table.size(1);
    index_t n_subspaces = table.size(0);
    TORCH_CHECK(d_code % BLOCK_SIZE == 0);
    TORCH_CHECK(n_queries % BLOCK_SIZE == 0);
    TORCH_CHECK(n_codewords % BLOCK_SIZE == 0);
    auto grad_query = torch::zeros_like(query);
    auto grad_table = torch::zeros_like(table);

    // dispatch
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n_codewords / BLOCK_SIZE, n_queries / BLOCK_SIZE, n_subspaces);
    AT_DISPATCH_FLOATING_TYPES(
        query.scalar_type(), "cdist_backward_kernel", ([&] {
            cdist_backward_kernel<scalar_t><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<scalar_t>(),
                table.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
                grad_query.data_ptr<scalar_t>(), grad_table.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return {grad_query, grad_table};
}