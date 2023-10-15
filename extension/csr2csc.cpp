#include "common.h"

std::vector<torch::Tensor> csr2csc_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values
) {
    CHECK_DIM(indptr, 2);
    CHECK_DIM(indices, 2);
    CHECK_DIM(values, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    CHECK_TYPE(values, torch::kFloat);
    TORCH_CHECK(indptr.size(0) == indices.size(0));
    TORCH_CHECK(indptr.size(0) == values.size(0));
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // sizes
    index_t n_rows = indptr.size(-1) - 1;
    index_t nonzeros = indices.size(-1);
    auto rev_indptr = torch::zeros_like(indptr);
    auto rev_indices = torch::zeros_like(indices);
    auto rev_values = torch::zeros_like(values);

    // cusparse
    for (auto i = 0; i < indptr.size(0); i += 1) {
        size_t external_size;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            handle, n_rows, n_rows, nonzeros,
            values.data_ptr<float>() + i * nonzeros,
            indptr.data_ptr<index_t>() + i * (n_rows + 1),
            indices.data_ptr<index_t>() + i * nonzeros,
            rev_values.data_ptr<float>() + i * nonzeros,
            rev_indptr.data_ptr<index_t>() + i * (n_rows + 1),
            rev_indices.data_ptr<index_t>() + i * nonzeros, CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, &external_size
        ));
        auto buffer = torch::zeros({external_size}, values.options());
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            handle, n_rows, n_rows, nonzeros,
            values.data_ptr<float>() + i * nonzeros,
            indptr.data_ptr<index_t>() + i * (n_rows + 1),
            indices.data_ptr<index_t>() + i * nonzeros,
            rev_values.data_ptr<float>() + i * nonzeros,
            rev_indptr.data_ptr<index_t>() + i * (n_rows + 1),
            rev_indices.data_ptr<index_t>() + i * nonzeros, CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, buffer.data_ptr<float>()
        ));
    }

    //
    return {rev_indptr, rev_indices, rev_values};
}
