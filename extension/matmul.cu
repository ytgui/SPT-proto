#include "common.h"

#define TM 8
#define TN 8
#define BK 16
#define BM 128
#define BN 128

// clang-format off
template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    index_t m, index_t n, index_t k, const scalar_t *left,
    const scalar_t *right, scalar_t *output
) {
    // index
    index_t thread_y = threadIdx.y;
    index_t thread_x = threadIdx.x;
    index_t global_y = blockIdx.y * BM;
    index_t global_x = blockIdx.x * BN;

    // cache
    __shared__ scalar_t cache_a[BM][BK];
    __shared__ scalar_t cache_b[BK][BN];

    // window
    scalar_t local_a[TM] = {0.0};
    scalar_t local_b[TN] = {0.0};
    scalar_t reduced[TM][TN] = {0.0};
    for (index_t offset_k = 0; offset_k < k; offset_k += BK) {
        // load
        for (index_t tile_y = 0; tile_y < TM; tile_y += 1) {
            index_t local_y = tile_y * blockDim.y + thread_y;
            cache_a[local_y][thread_x] = left[
                (global_y + local_y) * k + (offset_k + thread_x)
            ];
        }
        for (index_t tile_x = 0; tile_x < TN; tile_x += 1) {
            index_t local_x = tile_x * blockDim.x + thread_x;
            cache_b[thread_y][local_x] = right[
                (offset_k + thread_y) * n + (global_x + local_x)
            ];
        }
        __syncthreads();

        // reduce
        for (index_t i = 0; i < BK; i += 1) {
            // tile a
            for (index_t tile_y = 0; tile_y < TM; tile_y += 1) {
                index_t local_y = tile_y * blockDim.y + thread_y;
                local_a[tile_y] = cache_a[local_y][i];
            }
            // tile b
            for (index_t tile_x = 0; tile_x < TN; tile_x += 1) {
                index_t local_x = tile_x * blockDim.x + thread_x;
                local_b[tile_x] = cache_b[i][local_x];
            }
            // product
            for (index_t tile_y = 0; tile_y < TM; tile_y += 1) {
                for (index_t tile_x = 0; tile_x < TN; tile_x += 1) {
                    reduced[tile_y][tile_x] += local_a[tile_y] * local_b[tile_x];
                }
            }
        }
        __syncthreads();
    }

    // store
    for (index_t tile_y = 0; tile_y < TM; tile_y += 1) {
        for (index_t tile_x = 0; tile_x < TN; tile_x += 1) {
            index_t local_y = tile_y * blockDim.y + thread_y;
            index_t local_x = tile_x * blockDim.x + thread_x;
            output[
                (global_y + local_y) * n + (global_x + local_x)
            ] = reduced[tile_y][tile_x];
        }
    }
}
// clang-format on

torch::Tensor matmul_cuda(
    const torch::Tensor &left, const torch::Tensor &right
) {
    CHECK_DIM(left, 2);
    CHECK_DIM(right, 2);
    TORCH_CHECK(left.size(1) == right.size(0));
    TORCH_CHECK(left.scalar_type() == right.scalar_type());

    // sizes
    index_t m = left.size(0);
    index_t n = right.size(1);
    index_t k = left.size(1);
    TORCH_CHECK(k % BK == 0);
    TORCH_CHECK(n % BN == 0);
    TORCH_CHECK(m % BM == 0);
    auto output = torch::zeros({m, n}, left.options());

    // dispatch
    dim3 blocks(n / BN, m / BM);
    dim3 threads(BN / TN, BM / TM);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        left.scalar_type(), "matmul_cuda_kernel", ([&] {
            matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
                m, n, k, left.data_ptr<scalar_t>(), right.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return output;
}
