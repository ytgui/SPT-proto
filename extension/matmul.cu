#include "common.h"

#define BLOCK_SIZE 16

template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    index_t m, index_t n, index_t k, const scalar_t *left,
    const scalar_t *right, scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gx = blockIdx.x * blockDim.x + tx;

    // window
    scalar_t reduced = 0.0;
    for (index_t offset = 0; offset < k; offset += BLOCK_SIZE) {
        // cache
        __shared__ scalar_t cache_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t cache_b[BLOCK_SIZE][BLOCK_SIZE];

        // store
        cache_a[ty][tx] = left[gy * k + (offset + tx)];
        cache_b[ty][tx] = right[(offset + ty) * n + gx];
        __syncthreads();

        // product
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            reduced += cache_a[ty][i] * cache_b[i][tx];
        }
        __syncthreads();
    }

    // store
    output[gy * n + gx] = reduced;
}

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
    TORCH_CHECK(m % BLOCK_SIZE == 0);
    TORCH_CHECK(n % BLOCK_SIZE == 0);
    TORCH_CHECK(k % BLOCK_SIZE == 0);
    auto output = torch::zeros({m, n}, left.options());

    // dispatch
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / BLOCK_SIZE, m / BLOCK_SIZE);
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