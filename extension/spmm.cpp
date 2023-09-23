#include "common.h"

torch::Tensor spmm_forward_cuda(
    const torch::Tensor &trans_lhs, const torch::Tensor &trans_rhs,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values, const torch::Tensor &x
) {
    CHECK_DIM(x, 3);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 2);
    CHECK_DIM(values, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.size(0) == indices.size(0));
    TORCH_CHECK(indices.sizes() == values.sizes());
    TORCH_CHECK(x.scalar_type() == values.scalar_type());

    // sizes
    index_t d_head = x.size(-1);
    index_t batch_size = x.size(0);
    index_t seq_length = x.size(1);
    index_t nonzeros = indices.size(-1);
    TORCH_CHECK(indptr.size(-1) == seq_length + 1);
    auto output = torch::zeros_like(x);

    // format
    cusparseSpMatDescr_t lhs = {};
    cusparseDnMatDescr_t rhs = {}, target = {};
    CUSPARSE_CHECK(cusparseCreateCsr(
        &lhs, seq_length, seq_length, nonzeros, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), values.data_ptr<float>(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    CUSPARSE_CHECK(cusparseCsrSetStridedBatch(lhs, batch_size, 0, nonzeros));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &rhs, seq_length, d_head, d_head, x.data_ptr<float>(), CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(
        cusparseDnMatSetStridedBatch(rhs, batch_size, seq_length * d_head)
    );
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &target, seq_length, d_head, d_head, output.data_ptr<float>(),
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(
        cusparseDnMatSetStridedBatch(target, batch_size, seq_length * d_head)
    );
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // transpose
    auto op_lhs = (trans_lhs.item<bool>()) ? CUSPARSE_OPERATION_TRANSPOSE
                                           : CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto op_rhs = (trans_rhs.item<bool>()) ? CUSPARSE_OPERATION_TRANSPOSE
                                           : CUSPARSE_OPERATION_NON_TRANSPOSE;

    // cu_sparse
    size_t external_size;
    float alpha = 1.0, beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, op_lhs, op_rhs, &alpha, lhs, rhs, &beta, target, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &external_size
    ));
    auto buffer = torch::zeros({external_size}, x.options());
    CUSPARSE_CHECK(cusparseSpMM(
        handle, op_lhs, op_rhs, &alpha, lhs, rhs, &beta, target, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, buffer.data_ptr<float>()
    ));

    //
    return output;
}