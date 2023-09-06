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
    CHECK_DIM(x, 1);
    CHECK_CPU(indptr, 1);
    CHECK_CPU(indices, 1);
    CHECK_DIM(dense, 2);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.scalar_type() == dense.scalar_type());

    // sizes
    index_t d_model = x.size(0);
    index_t nonzeros = indices.size(-1);
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // blocks
    TORCH_CHECK(config.dim() == 3);
    index_t in_blocks = config.size(1);
    index_t out_blocks = config.size(0);
    index_t block_size = config.size(-1);
    index_t block_stride = in_blocks * block_size;
    TORCH_CHECK(indptr.size(0) == out_blocks + 1);
    TORCH_CHECK(dense.size(1) == in_blocks * block_size);
    TORCH_CHECK(dense.size(0) == out_blocks * block_size);
    auto output = torch::zeros({out_blocks * block_size}, x.options());

    //
    float alpha = 1.0, beta = 1.0;
    const auto x_ptr = x.data_ptr<float>();
    const auto y_ptr = output.data_ptr<float>();
    const auto dense_ptr = dense.data_ptr<float>();
    const auto indptr_ptr = indptr.accessor<index_t, 1>();
    const auto indices_ptr = indices.accessor<index_t, 1>();
    for (auto row = 0; row < out_blocks; row += 1) {
        const auto y_i = &y_ptr[row * block_size];
        for (index_t i = indptr_ptr[row]; i < indptr_ptr[row + 1]; i += 1) {
            index_t col = indices_ptr[i];
            //
            const auto x_i = &x_ptr[col * block_size];
            const auto w_i = &dense_ptr[
                row * block_size * block_stride + col * block_size
            ];
            CUBLAS_CHECK(cublasSgemv(
                handle, CUBLAS_OP_T, block_size, block_size, &alpha, w_i,
                block_stride, x_i, 1, &beta, y_i, 1
            ));
            // CUDA_CHECH(cudaDeviceSynchronize());
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
    CHECK_DIM(x, 1);
    CHECK_CPU(indptr, 1);
    CHECK_CPU(indices, 1);
    CHECK_DIM(dense, 2);
    CHECK_DIM(grad_output, 1);
    CHECK_TYPE(indptr, torch::kInt32);
    CHECK_TYPE(indices, torch::kInt32);
    TORCH_CHECK(x.scalar_type() == dense.scalar_type());
    TORCH_CHECK(x.scalar_type() == grad_output.scalar_type());

    // sizes
    index_t d_model = x.size(0);
    index_t nonzeros = indices.size(-1);
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // blocks
    TORCH_CHECK(config.dim() == 3);
    index_t in_blocks = config.size(1);
    index_t out_blocks = config.size(0);
    index_t block_size = config.size(-1);
    index_t block_stride = in_blocks * block_size;
    TORCH_CHECK(indptr.size(0) == out_blocks + 1);
    TORCH_CHECK(dense.size(1) == in_blocks * block_size);
    TORCH_CHECK(dense.size(0) == out_blocks * block_size);
    TORCH_CHECK(grad_output.size(0) == out_blocks * block_size);
    auto grad_weight = torch::zeros_like(dense);
    auto grad_x = torch::zeros_like(x);

    //
    float alpha = 1.0, beta = 1.0;
    const auto x_ptr = x.data_ptr<float>();
    const auto dense_ptr = dense.data_ptr<float>();
    const auto grad_x_ptr = grad_x.data_ptr<float>();
    const auto grad_weight_ptr = grad_weight.data_ptr<float>();
    const auto grad_output_ptr = grad_output.data_ptr<float>();
    const auto indptr_ptr = indptr.accessor<index_t, 1>();
    const auto indices_ptr = indices.accessor<index_t, 1>();
    for (auto row = 0; row < out_blocks; row += 1) {
        const auto grad_y_i = &grad_output_ptr[row * block_size];
        for (index_t i = indptr_ptr[row]; i < indptr_ptr[row + 1]; i += 1) {
            index_t col = indices_ptr[i];
            //
            const auto w_i = &dense_ptr[
                row * block_size * block_stride + col * block_size
            ];
            const auto grad_x_i = &grad_x_ptr[col * block_size];
            CUBLAS_CHECK(cublasSgemv(
                handle, CUBLAS_OP_N, block_size, block_size, &alpha, w_i,
                block_stride, grad_y_i, 1, &beta, grad_x_i, 1
            ));
            // CUDA_CHECH(cudaDeviceSynchronize());
        }
    }

    //
    return {grad_weight, grad_x};
}