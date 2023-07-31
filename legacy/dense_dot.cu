template <typename scalar_t>
__global__ void sparse_mha_forward_cuda(
    index_t seq_length, index_t d_head, const scalar_t *q, const scalar_t *k,
    scalar_t *output
) {
    // index
    index_t ty = threadIdx.y;
    index_t tx = threadIdx.x;
    index_t gy = blockIdx.y * blockDim.y + ty;
    index_t gx = blockIdx.x * blockDim.x + tx;

    // window
    scalar_t reduced = 0.0;
    for (index_t offset = 0; offset < d_head; offset += BLOCK_SIZE) {
        // cache
        __shared__ scalar_t cache_q[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ scalar_t cache_k[BLOCK_SIZE][BLOCK_SIZE];

        // store
        // a[gy, tx], b[ty, gx], bT[gx, ty]
        cache_q[ty][tx] = q[gy * d_head + (offset + tx)];
        cache_k[ty][tx] = k[gx * d_head + (offset + ty)];
        __syncthreads();

        // product
        for (index_t i = 0; i < BLOCK_SIZE; i += 1) {
            reduced += cache_q[ty][i] * cache_k[i][tx];
        }
    }

    // store
    output[gy * seq_length + gx] = reduced;
}