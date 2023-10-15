#include "common.h"

std::vector<torch::Tensor> csr2csc_cuda(
    const torch::Tensor &config, const torch::Tensor &indptr,
    const torch::Tensor &indices, const torch::Tensor &values
) {
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_DIM(values, 1);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    CHECK_TYPE(values, torch::kFloat);
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // sizes
    index_t n_cols = config.size(0);
    index_t n_rows = indptr.size(-1) - 1;
    index_t nonzeros = indices.size(-1);
    auto rev_indptr = torch::zeros_like(indptr);
    auto rev_indices = torch::zeros_like(indices);
    auto rev_values = torch::zeros_like(values);

    // cusparse
    size_t external_size;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, n_rows, n_cols, nonzeros, values.data_ptr<float>(),
        indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
        rev_values.data_ptr<float>(), rev_indptr.data_ptr<index_t>(),
        rev_indices.data_ptr<index_t>(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &external_size
    ));
    auto buffer = torch::zeros({external_size}, values.options());
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, n_rows, n_cols, nonzeros, values.data_ptr<float>(),
        indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
        rev_values.data_ptr<float>(), rev_indptr.data_ptr<index_t>(),
        rev_indices.data_ptr<index_t>(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        buffer.data_ptr<float>()
    ));

    //
    return {rev_indptr, rev_indices, rev_values};
}
