#include "common.h"

#define TSZ 4
#define BSZ 16

// clang-format off
template <typename scalar_t, typename vector_t, unsigned K>
__global__ void cdist_forward_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table,
    scalar_t *distance, index_t *indices) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // load query
    __shared__ scalar_t cache_query[BSZ][K];
    for (index_t i = 0; i < K; i += TSZ) {
        *(vector_t *)&cache_query[ty][i] = __ldg(
            (const vector_t *)&query[
                gz * n_queries * d_code + gy * d_code + i
            ]
        );
    }

    // window
    index_t min_index = 0;
    scalar_t min_distance = 1e13;
    for (index_t offset_x = 0; offset_x < n_codewords; offset_x += BSZ) {
        // load table
        __shared__ scalar_t cache_table[BSZ][K];
        for (index_t i = 0; i < K; i += TSZ) {
            *(vector_t *)&cache_table[ty][i] = __ldg(
                (const vector_t *)&table[
                    gz * n_codewords * d_code + (offset_x + ty) * d_code + i
                ]
            );
        }
        __syncthreads();

        // distance
        for (index_t local_x = 0; local_x < BSZ; local_x += TSZ) {
            // reduce
            scalar_t reduced[TSZ] = {};
            for (index_t t = 0; t < TSZ; t += 1) {
                for (index_t i = 0; i < K; i += 1) {
                    reduced[t] += fabsf(
                        cache_query[ty][i] - cache_table[local_x + t][i]
                    );
                }
                bool cond = reduced[t] < min_distance;
                min_index = cond ? (offset_x + local_x + t) : min_index;
                min_distance = cond ? reduced[t] : min_distance;
            }
            // store
            index_t offset_b = gz * n_queries * n_codewords;
            __stcs(
                (vector_t *)&distance[
                    offset_b + gy * n_codewords + (offset_x + local_x)
                ], *(const vector_t *)reduced
            );
        }
        __syncthreads();
    }

    // store indices
    indices[gz * n_queries + gy] = min_index;
}

template <typename scalar_t, typename vector_t, unsigned K>
__global__ void cdist_backward_query_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, const scalar_t *grad_output,
    scalar_t *grad_query) {
    // index
    index_t ty = threadIdx.y;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;

    // load query
    __shared__ scalar_t cache_query[BSZ][K];
    for (index_t i = 0; i < K; i += TSZ) {
        *(vector_t *)&cache_query[ty][i] = __ldg(
            (const vector_t *)&query[
                gz * n_queries * d_code + gy * d_code + i
            ]
        );
    }

    // window
    scalar_t reduced[K] = {};
    for (index_t offset_x = 0; offset_x < n_codewords; offset_x += BSZ) {
        // load table
        __shared__ scalar_t cache_table[BSZ][K];
        for (index_t i = 0; i < K; i += TSZ) {
            *(vector_t *)&cache_table[ty][i] = __ldg(
                (const vector_t *)&table[
                    gz * n_codewords * d_code + (offset_x + ty) * d_code + i
                ]
            );
        }
        __shared__ scalar_t cache_grad_output[BSZ][BSZ];
        for (index_t local_x = 0; local_x < BSZ; local_x += TSZ) {
            *(vector_t *)&cache_grad_output[ty][local_x] = __ldg(
                (const vector_t *)&grad_output[
                    gz * n_queries * n_codewords + gy * n_codewords + (offset_x + local_x)
                ]
            );
        }
        __syncthreads();

        // product
        for (index_t local_x = 0; local_x < BSZ; local_x += 1) {
            scalar_t grad_v = cache_grad_output[ty][local_x];
            for (index_t i = 0; i < K; i += 1) {
                reduced[i] += (cache_query[ty][i] - cache_table[local_x][i]) > 0 ? grad_v : -grad_v;
            }
        }
        __syncthreads();
    }

    // store
    for (index_t i = 0; i < K; i += TSZ) {
        __stcs(
            (vector_t *)&grad_query[
                gz * n_queries * d_code + gy * d_code + i
            ], *(const vector_t *)&reduced[i]
        );
    }
}

template <typename scalar_t, typename vector_t>
__global__ void cdist_backward_table_kernel(
    index_t n_queries, index_t n_codewords, index_t d_code,
    const scalar_t *query, const scalar_t *table, const scalar_t *grad_output,
    scalar_t *grad_table) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gz = blockIdx.z * blockDim.z;
    index_t gy = blockIdx.y * blockDim.y + ty;  // k
    index_t gx = blockIdx.x * blockDim.x + tx;  // n

    // cache
    __shared__ scalar_t cache_table[TSZ][BSZ];
    cache_table[ty][tx] = table[
        gz * n_codewords * d_code + gx * d_code + gy
    ];

    // window
    scalar_t reduced = 0.0;
    for (index_t offset_y = 0; offset_y < n_queries; offset_y += BSZ) {
        // cache
        __shared__ scalar_t cache_query[BSZ][TSZ];
        __shared__ scalar_t cache_grad_output[BSZ][BSZ];
        cache_query[tx][ty] = query[
            gz * n_queries * d_code + (offset_y + tx) * d_code + gy
        ];
        for (index_t i = 0; i < BSZ / TSZ; i += 1) {
            cache_grad_output[i * TSZ + ty][tx] = grad_output[
                gz * n_queries * n_codewords + (offset_y + i * TSZ + ty) * n_codewords + gx
            ];
        }
        __syncthreads();

        // product
        for (index_t i = 0; i < BSZ; i += 1) {
            scalar_t grad_v = cache_grad_output[i][tx];
            scalar_t grad_abs = (
                cache_query[i][ty] - cache_table[ty][tx]
            ) > 0 ? grad_v : -grad_v;
            reduced -= grad_abs;
        }
        __syncthreads();
    }

    // store
    grad_table[
        gz * n_codewords * d_code + gx * d_code + gy
    ] = reduced;
}
// clang-format on

std::vector<torch::Tensor> cdist_forward_cuda(
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
    auto distance = torch::empty(
        {n_subspaces, n_queries, n_codewords}, query.options()
    );
    auto indices = torch::empty(
        {n_subspaces, n_queries}, query.options().dtype(torch::kInt32)
    );

    // dispatch
    dim3 threads(1, BSZ);
    dim3 blocks(1, n_queries / BSZ, n_subspaces);
    if (d_code == 4) {
        cdist_forward_kernel<float, float4, 4><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), distance.data_ptr<float>(),
            indices.data_ptr<index_t>()
        );
    } else if (d_code == 8) {
        cdist_forward_kernel<float, float4, 8><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), distance.data_ptr<float>(),
            indices.data_ptr<index_t>()
        );
    } else if (d_code == 16) {
        cdist_forward_kernel<float, float4, 16><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), distance.data_ptr<float>(),
            indices.data_ptr<index_t>()
        );
    } else if (d_code == 24) {
        cdist_forward_kernel<float, float4, 24><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), distance.data_ptr<float>(),
            indices.data_ptr<index_t>()
        );
    } else if (d_code == 32) {
        cdist_forward_kernel<float, float4, 32><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), distance.data_ptr<float>(),
            indices.data_ptr<index_t>()
        );
    } else {
        TORCH_CHECK(false && "d_code not supported");
    }
    CUDA_CHECH(cudaGetLastError());

    //
    return {distance, indices};
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

    // dispatch query
    [&]() {
        dim3 threads(1, BSZ);
        dim3 blocks(1, n_queries / BSZ, n_subspaces);
        if (d_code == 4) {
            cdist_backward_query_kernel<float, float4, 4><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<float>(),
                table.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_query.data_ptr<float>()
            );
        } else if (d_code == 8) {
            cdist_backward_query_kernel<float, float4, 8><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<float>(),
                table.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_query.data_ptr<float>()
            );
        } else if (d_code == 16) {
            cdist_backward_query_kernel<float, float4, 16><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<float>(),
                table.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_query.data_ptr<float>()
            );
        } else if (d_code == 24) {
            cdist_backward_query_kernel<float, float4, 24><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<float>(),
                table.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_query.data_ptr<float>()
            );
        } else if (d_code == 32) {
            cdist_backward_query_kernel<float, float4, 32><<<blocks, threads>>>(
                n_queries, n_codewords, d_code, query.data_ptr<float>(),
                table.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_query.data_ptr<float>()
            );
        } else {
            TORCH_CHECK(false && "d_code not supported");
        }
        CUDA_CHECH(cudaGetLastError());
    }();

    // dispatch table
    [&]() {
        dim3 threads(BSZ, TSZ);
        dim3 blocks(n_codewords / BSZ, d_code / TSZ, n_subspaces);
        cdist_backward_table_kernel<float, float4><<<blocks, threads>>>(
            n_queries, n_codewords, d_code, query.data_ptr<float>(),
            table.data_ptr<float>(), grad_output.data_ptr<float>(),
            grad_table.data_ptr<float>()
        );
        CUDA_CHECH(cudaGetLastError());
    }();

    //
    return {grad_query, grad_table};
}