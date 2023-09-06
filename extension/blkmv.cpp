#include "common.h"

#define CHECK_CPU(x, d)                                                   \
    TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")                   \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

torch::Tensor blkmv_forward_cuda(
    const torch::Tensor &config, const torch::Tensor &dense,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &x
) {
    CHECK_DIM(x, 2);
    CHECK_CPU(indptr, 2);
    CHECK_CPU(indices, 2);
    CHECK_DIM(dense, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.size(0) == indptr.size(0));
    TORCH_CHECK(x.size(0) == indices.size(0));
    TORCH_CHECK(x.scalar_type() == dense.scalar_type());

    // sizes
    index_t batch_size = x.size(0);
    index_t in_features = x.size(-1);
    index_t nonzeros = indices.size(-1);
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // blocks
    TORCH_CHECK(config.dim() == 3);
    index_t in_blocks = config.size(1);
    index_t out_blocks = config.size(0);
    index_t block_size = config.size(-1);
    index_t block_stride = in_blocks * block_size;
    index_t out_features = out_blocks * block_size;
    TORCH_CHECK(indptr.size(-1) == out_blocks + 1);
    TORCH_CHECK(dense.size(1) == in_blocks * block_size);
    TORCH_CHECK(dense.size(0) == out_blocks * block_size);
    auto output = torch::zeros({batch_size, out_features}, x.options());

    //
    float alpha = 1.0, beta = 1.0;
    const auto x_ptr = x.data_ptr<float>();
    const auto y_ptr = output.data_ptr<float>();
    const auto dense_ptr = dense.data_ptr<float>();
    const auto indptr_ptr = indptr.accessor<index_t, 2>();
    const auto indices_ptr = indices.accessor<index_t, 2>();
    for (auto b = 0; b < batch_size; b += 1) {
        for (auto row = 0; row < out_blocks; row += 1) {
            const auto y_i = &y_ptr[b * out_features + row * block_size];
            for (index_t i = indptr_ptr[b][row]; i < indptr_ptr[b][row + 1];
                 i += 1) {
                index_t col = indices_ptr[b][i];
                //
                const auto x_i = &x_ptr[b * in_features + col * block_size];
                const auto w_i =
                    &dense_ptr
                        [row * block_size * block_stride + col * block_size];
                CUBLAS_CHECK(cublasSgemv(
                    handle, CUBLAS_OP_T, block_size, block_size, &alpha, w_i,
                    block_stride, x_i, 1, &beta, y_i, 1
                ));
            }
        }
    }

    //
    return output;
}

std::vector<torch::Tensor> blkmv_backward_cuda(
    const torch::Tensor &config, const torch::Tensor &dense,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &x, const torch::Tensor &grad_output
) {
    CHECK_DIM(x, 2);
    CHECK_CPU(indptr, 2);
    CHECK_CPU(indices, 2);
    CHECK_DIM(dense, 2);
    CHECK_DIM(grad_output, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.size(0) == indptr.size(0));
    TORCH_CHECK(x.size(0) == indices.size(0));
    TORCH_CHECK(x.size(0) == grad_output.size(0));
    TORCH_CHECK(x.scalar_type() == dense.scalar_type());
    TORCH_CHECK(x.scalar_type() == grad_output.scalar_type());

    // sizes
    index_t batch_size = x.size(0);
    index_t in_features = x.size(-1);
    index_t nonzeros = indices.size(-1);
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // blocks
    TORCH_CHECK(config.dim() == 3);
    index_t in_blocks = config.size(1);
    index_t out_blocks = config.size(0);
    index_t block_size = config.size(-1);
    index_t block_stride = in_blocks * block_size;
    index_t out_features = out_blocks * block_size;
    TORCH_CHECK(dense.size(1) == in_features);
    TORCH_CHECK(dense.size(0) == out_features);
    TORCH_CHECK(indptr.size(-1) == out_blocks + 1);
    TORCH_CHECK(grad_output.size(-1) == out_features);
    auto grad_weight = torch::zeros_like(dense);
    auto grad_x = torch::zeros_like(x);

    //
    float alpha = 1.0, beta = 1.0;
    const auto x_ptr = x.data_ptr<float>();
    const auto dense_ptr = dense.data_ptr<float>();
    const auto grad_x_ptr = grad_x.data_ptr<float>();
    const auto grad_weight_ptr = grad_weight.data_ptr<float>();
    const auto grad_output_ptr = grad_output.data_ptr<float>();
    const auto indptr_ptr = indptr.accessor<index_t, 2>();
    const auto indices_ptr = indices.accessor<index_t, 2>();
    for (auto b = 0; b < batch_size; b += 1) {
        for (auto row = 0; row < out_blocks; row += 1) {
            const auto grad_y_i = &grad_output_ptr[b * out_features + row * block_size];
            for (index_t i = indptr_ptr[b][row]; i < indptr_ptr[b][row + 1]; i += 1) {
                index_t col = indices_ptr[b][i];
                //
                const auto w_i = &dense_ptr[
                    row * block_size * block_stride + col * block_size
                ];
                const auto grad_x_i = &grad_x_ptr[
                    b * in_features + col * block_size
                ];
                CUBLAS_CHECK(cublasSgemv(
                    handle, CUBLAS_OP_N, block_size, block_size, &alpha, w_i,
                    block_stride, grad_y_i, 1, &beta, grad_x_i, 1
                ));
            }
        }
    }

    //
    auto device = x.device();
    for (auto row = 0; row < out_blocks; row += 1) {
        std::vector<std::vector<index_t>> batch_list(in_blocks);
        for (auto b = 0; b < batch_size; b += 1) {
            for (index_t i = indptr_ptr[b][row]; i < indptr_ptr[b][row + 1]; i += 1) {
                index_t col = indices_ptr[b][i];
                batch_list[col].push_back(b);
            }
        }
        //
        for (index_t col = 0; col < in_blocks; col += 1) {
            if (batch_list[col].size() <= 0) {
                continue;
            }
            auto index = torch::tensor(batch_list[col]).to(device);
            auto x_slice = torch::index_select(x, /*dim=*/ 0, index);
            auto grad_slice = torch::index_select(grad_output, /*dim=*/ 0, index);
            //
            x_slice = x_slice.index(
                {torch::indexing::Slice(), torch::indexing::Slice(col * block_size, (col + 1) * block_size)}
            );
            grad_slice = grad_slice.index(
                {torch::indexing::Slice(), torch::indexing::Slice(row * block_size, (row + 1) * block_size)}
            );
            grad_weight.index_put_(
                {torch::indexing::Slice(row * block_size, (row + 1) * block_size),
                 torch::indexing::Slice(col * block_size, (col + 1) * block_size)},
                 torch::matmul(grad_slice.t(), x_slice)
            );
        }
    }

    //
    return {grad_weight, grad_x};
}