#include "common.h"

#define BK 8
#define BM 64
#define BN 64
#define TN BK

template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    index_t m, index_t n, index_t k, const scalar_t *left,
    const scalar_t *right, scalar_t *output
) {
    // cache
    __shared__ scalar_t cache_a[BM][BK];
    __shared__ scalar_t cache_b[BK][BN];

    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t by = blockIdx.y;
    index_t bx = blockIdx.x;

    // window
    scalar_t reduced[TN] = {0.0};
    for (index_t offset = 0; offset < k; offset += BK) {
        index_t row = by * BM + ty;
        index_t col = bx * BN + ty;
        cache_a[ty][tx] = left[row * k + (offset + tx)];
        cache_b[tx][ty] = right[(offset + tx) * n + col];
        __syncthreads();

        // reduce
        #pragma unroll
        for (index_t i = 0; i < BK; i += 1) {
            const scalar_t const_a = cache_a[ty][i];

            // tile
            #pragma unroll
            for (index_t t = 0; t < TN; t += 1) {
                reduced[t] += const_a * cache_b[i][t * TN + tx];
            }
        }
        __syncthreads();
    }

    // store
    for (index_t t = 0; t < TN; t += 1) {
        index_t row = by * BM + ty;
        index_t col = bx * BN + t * TN + tx;
        output[row * n + col] = reduced[t];
    }
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
    TORCH_CHECK(k % BK == 0);
    TORCH_CHECK(n % BN == 0);
    TORCH_CHECK(m % BM == 0);
    auto output = torch::zeros({m, n}, left.options());

    // dispatch
    dim3 threads(BK, BM);
    dim3 blocks(n / BN, m / BM);
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
