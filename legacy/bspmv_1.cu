#include "common.h"

#define TSZ 4
#define BSZ 64

template <typename scalar_t>
__global__ void bspmv_forward_kernel(
    index_t m, index_t n, index_t k, const scalar_t *left,
    const scalar_t *right, scalar_t *output
) {
    // index
    index_t thread_y = threadIdx.y;
    index_t thread_x = threadIdx.x;
    index_t global_y = blockIdx.y * BSZ;

    // cache
    __shared__ scalar_t cache_lhs[BSZ][BSZ];
    __shared__ scalar_t cache_rhs[BSZ][BSZ];

    // window
    for (index_t offset_n = 0; offset_n < n; offset_n += BSZ) {
        // reduce
        scalar_t reduced[TSZ][TSZ] = {};

        // feature
        for (index_t offset_k = 0; offset_k < k; offset_k += BSZ) {
            // load
            for (index_t tile_y = 0; tile_y < TSZ; tile_y += 1) {
                index_t local_y = thread_y * TSZ + tile_y;
                for (index_t tile_x = 0; tile_x < TSZ; tile_x += 1) {
                    index_t local_x = thread_x * TSZ + tile_x;
                    cache_lhs[local_y][local_x] = left[
                        (global_y + local_y) * k + (offset_k + local_x)
                    ];
                    cache_rhs[local_y][local_x] = right[
                        (offset_n + local_y) * k + (offset_k + local_x)
                    ];
                }
            }
            __syncthreads();

            // product
            for (index_t tile_y = 0; tile_y < TSZ; tile_y += 1) {
                index_t local_y = thread_y * TSZ + tile_y;
                for (index_t tile_x = 0; tile_x < TSZ; tile_x += 1) {
                    index_t local_x = thread_x * TSZ + tile_x;
                    for (index_t i = 0; i < BSZ; i += 1) {
                        reduced[tile_y][tile_x] += cache_lhs[local_y][i] * cache_rhs[local_x][i];
                    }
                }
            }
            __syncthreads();
        }

        // store
        for (index_t tile_y = 0; tile_y < TSZ; tile_y += 1) {
            index_t local_y = thread_y * TSZ + tile_y;
            for (index_t tile_x = 0; tile_x < TSZ; tile_x += 1) {
                index_t local_x = thread_x * TSZ + tile_x;
                output[(global_y + local_y) * n + (offset_n + local_x)] = reduced[tile_y][tile_x];
            }
        }
    }
}

torch::Tensor bspmv_forward_cuda(
    const torch::Tensor &left, const torch::Tensor &right
) {
    CHECK_DIM(left, 2);
    CHECK_DIM(right, 2);
    TORCH_CHECK(left.size(-1) == right.size(-1));
    TORCH_CHECK(left.scalar_type() == right.scalar_type());

    // sizes
    index_t m = left.size(0);
    index_t k = left.size(1);
    index_t n = right.size(0);
    TORCH_CHECK(m % BSZ == 0);
    TORCH_CHECK(n % BSZ == 0);
    TORCH_CHECK(k % TSZ == 0);
    auto output = torch::zeros({m, n}, left.options());

    // dispatch
    dim3 blocks(1, m / BSZ);
    dim3 threads(BSZ / TSZ, BSZ / TSZ);
    bspmv_forward_kernel<float><<<blocks, threads>>>(
        m, n, k, left.data_ptr<float>(), right.data_ptr<float>(),
        output.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);

    //
    return output;
}