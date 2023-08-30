#include "common.h"

#define BK 16
#define BM 16

template <typename scalar_t>
__global__ void sddmm_forward_cuda_kernel(
    index_t seq_length, index_t d_head, const index_t *indptr,
    const index_t *indices, const scalar_t *query, const scalar_t *key,
    scalar_t *output
) {
    // index
    index_t local_y = threadIdx.x;
    index_t global_y = blockIdx.x * blockDim.x;
    index_t row = global_y + local_y;

    // contract
    for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
        index_t col = indices[i];

        // product
        scalar_t reduced = 0.0;
        for (index_t k = 0; k < d_head; k += 1) {
            reduced += query[row * d_head + k] * key[col * d_head + k];
        }

        // store
        output[i] = reduced;
    }
}

torch::Tensor sddmm_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_DIM(key, 2);
    CHECK_DIM(query, 2);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_TYPE(indptr, torch::kInt64);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    TORCH_CHECK((d_head % BK) == 0);
    index_t seq_length = query.size(0);
    TORCH_CHECK((seq_length % BM) == 0);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // dispatch
    dim3 threads(BM);
    dim3 blocks(seq_length / BM);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "sddmm_forward_cuda_kernel", ([&] {
            sddmm_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                seq_length, d_head, indptr.data_ptr<index_t>(),
                indices.data_ptr<index_t>(), query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(), output.data_ptr<scalar_t>()
            );
            TORCH_CHECK(cudaGetLastError() == cudaSuccess);
        })
    );

    //
    return output;
}