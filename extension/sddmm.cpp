#include "common.h"

torch::Tensor sddmm_forward_cuda(
    const torch::Tensor &trans_lhs, const torch::Tensor &trans_rhs,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
) {
    CHECK_DIM(key, 3);
    CHECK_DIM(query, 3);
    CHECK_DIM(indptr, 1);
    CHECK_DIM(indices, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(query.sizes() == key.sizes());
    TORCH_CHECK(query.size(0) == indices.size(0));
    TORCH_CHECK(query.scalar_type() == key.scalar_type());

    // sizes
    index_t d_head = query.size(-1);
    index_t batch_size = query.size(0);
    index_t seq_length = query.size(1);
    index_t nonzeros = indices.size(-1);
    TORCH_CHECK(indptr.size(-1) == seq_length + 1);
    auto output = torch::zeros_like(indices, query.options());

    // format
    cusparseSpMatDescr_t target = {};
    cusparseDnMatDescr_t lhs = {}, rhs = {};
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &lhs, seq_length, d_head, d_head, query.data_ptr<float>(), CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(
        cusparseDnMatSetStridedBatch(lhs, batch_size, seq_length * d_head)
    );
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &rhs, seq_length, d_head, d_head, key.data_ptr<float>(), CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(
        cusparseDnMatSetStridedBatch(rhs, batch_size, seq_length * d_head)
    );
    CUSPARSE_CHECK(cusparseCreateCsr(
        &target, seq_length, seq_length, nonzeros, indptr.data_ptr<index_t>(),
        indices.data_ptr<index_t>(), output.data_ptr<float>(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    CUSPARSE_CHECK(cusparseCsrSetStridedBatch(target, batch_size, 0, nonzeros));
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // transpose
    auto op_lhs = (trans_lhs.item<bool>()) ? CUSPARSE_OPERATION_TRANSPOSE
                                           : CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto op_rhs = (trans_rhs.item<bool>()) ? CUSPARSE_OPERATION_TRANSPOSE
                                           : CUSPARSE_OPERATION_NON_TRANSPOSE;

    // cu_sparse
    size_t external_size;
    float alpha = 1.0, beta = 0.0;
    CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
        handle, op_lhs, op_rhs, &alpha, lhs, rhs, &beta, target, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &external_size
    ));
    auto buffer = torch::zeros({external_size}, query.options());
    CUSPARSE_CHECK(cusparseSDDMM(
        handle, op_lhs, op_rhs, &alpha, lhs, rhs, &beta, target, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buffer.data_ptr<float>()
    ));

    //
    return output;
}
