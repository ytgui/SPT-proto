/*
template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *values,
    const scalar_t *x, scalar_t *output
) {
    // index
    index_t i = threadIdx.x;
    index_t h = threadIdx.y;
    index_t row = blockIdx.x;
    index_t n = blockIdx.y;
    index_t sp_offset = n * n_nonzeros * n_heads;
    index_t qk_offset = n * seq_length * n_heads * d_head;

    // product
    scalar_t reduced = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];
        reduced += values[sp_offset + cursor * n_heads + h] *
                   x[qk_offset + col * n_heads * d_head + h * d_head + i];
    }

    // store
    output[qk_offset + row * n_heads * d_head + h * d_head + i] = reduced;
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    index_t n_nonzeros, index_t seq_length, index_t n_heads, index_t d_head,
    const index_t *indptr, const index_t *indices, const scalar_t *values,
    const scalar_t *x, const scalar_t *grad_output, value_t *grad_a,
    value_t *grad_x
) {
    // index
    index_t i = threadIdx.x;
    index_t h = threadIdx.y;
    index_t row = blockIdx.x;
    index_t n = blockIdx.y;
    index_t sp_offset = n * n_nonzeros * n_heads;
    index_t qk_offset = n * seq_length * n_heads * d_head;

    // product
    scalar_t reduced = 0.0;
    for (index_t cursor = indptr[row]; cursor < indptr[row + 1]; cursor += 1) {
        index_t col = indices[sp_offset + cursor * n_heads + h];
        reduced += values[sp_offset + cursor * n_heads + h] *
                   x[qk_offset + col * n_heads * d_head + h * d_head + i];
    }

    // store
    output[qk_offset + row * n_heads * d_head + h * d_head + i] = reduced;
}
*/
