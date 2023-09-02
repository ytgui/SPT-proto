#include "common.h"

torch::Tensor spmm_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values, const torch::Tensor &x
) {
    CHECK_DIM(x, 2);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 1);
    CHECK_DIM(values, 1);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.scalar_type() == values.scalar_type());

    // sizes
    index_t d_head = x.size(-1);
    index_t seq_length = x.size(0);
    TORCH_CHECK(indptr.size(0) == seq_length + 1);
    auto output = torch::zeros_like(x, x.options());

    // format
    cusparseSpMatDescr_t lhs;
    cusparseDnMatDescr_t rhs, target;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &lhs, seq_length, seq_length, indices.size(0),
        indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),
        values.data_ptr<float>(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &rhs, x.size(0), x.size(-1), x.size(-1), x.data_ptr<float>(),
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &target, output.size(0), output.size(-1), output.size(-1),
        output.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW
    ));
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // cu_sparse
    size_t external_size;
    float alpha = 1.0, beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs, rhs, &beta, target,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &external_size
    ));
    auto buffer = torch::zeros({external_size}, x.options());
    CUSPARSE_CHECK(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lhs, rhs, &beta, target,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer.data_ptr<float>()
    ));

    //
    return output;
}