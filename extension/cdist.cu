#include "common.h"

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
        // reduce
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

template <typename scalar_t, typename vector_t>
__global__ void cdist_backward_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, const scalar_t *grad_output,
    scalar_t *grad_query, scalar_t *grad_table) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // window
    for (index_t offset_k = 0; offset_k < d_code; offset_k += TSZ) {
        //cache
        __shared__ vector_t cache_a[BSZ];
        cache_a[ty] = __ldg(
            (const vector_t *)&query[
                gz * n_queries * d_code + gy * d_code + offset_k
            ]
        );

        // reduce
        vector_t reduced = {};
        for (index_t offset_x = 0; offset_x < n_codewords; offset_x += BSZ) {
            // cache
            __shared__ vector_t cache_b[BSZ];
            __shared__ scalar_t cache_grad[BSZ][BSZ];
            cache_b[ty] = __ldg(
                (const vector_t *)&table[
                    gz * n_codewords * d_code + (offset_x + ty) * d_code + offset_k
                ]
            );
            for (index_t tx = 0; tx < BSZ; tx += TSZ) {
                *(vector_t *)&cache_grad[ty][tx] = __ldg(
                    (const vector_t *)&grad_output[
                        gz * n_queries * n_codewords + gy * n_codewords + (offset_x + tx)
                    ]
                );
            }
            __syncthreads();

            // product
            for (index_t tx = 0; tx < BSZ; tx += 1) {
                scalar_t grad_v = cache_grad[ty][tx];
                vector_t grad_abs = {
                    (cache_a[ty].x - cache_b[tx].x) > 0 ? grad_v : -grad_v,
                    (cache_a[ty].y - cache_b[tx].y) > 0 ? grad_v : -grad_v,
                    (cache_a[ty].z - cache_b[tx].z) > 0 ? grad_v : -grad_v,
                    (cache_a[ty].w - cache_b[tx].w) > 0 ? grad_v : -grad_v,
                };
                reduced.x += grad_abs.x;
                reduced.y += grad_abs.y;
                reduced.z += grad_abs.z;
                reduced.w += grad_abs.w;
            }
            __syncthreads();
        }

        // store
        __stcs(
            (vector_t *)&grad_query[
                gz * n_queries * d_code + gy * d_code + offset_k
            ], reduced
        );
    }
}
// clang-format on

torch::Tensor cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
) {
    CHECK_DIM(query, 3);
    CHECK_DIM(table, 3);
    CHECK_TYPE(query, torch::kFloat32);
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
    CHECK_TYPE(query, torch::kFloat32);
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
    TORCH_CHECK(d_code % TSZ == 0);
    TORCH_CHECK(n_queries % BSZ == 0);
    TORCH_CHECK(n_codewords % BSZ == 0);
    auto grad_query = torch::zeros_like(query);
    auto grad_table = torch::zeros_like(table);

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, n_queries / BSZ, n_subspaces);
    cdist_backward_kernel<float, float4><<<blocks, threads>>>(
        n_queries, n_codewords, d_code, query.data_ptr<float>(),
        table.data_ptr<float>(), grad_output.data_ptr<float>(),
        grad_query.data_ptr<float>(), grad_table.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return {grad_query, grad_table};
}