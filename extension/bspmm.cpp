#include "common.h"

#define CHECK_CPU(x, d)                                                   \
    TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")                   \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

torch::Tensor bspmm_forward_cuda(
    const torch::Tensor &x, const torch::Tensor &weight,
    const torch::Tensor &indices

) {
    CHECK_DIM(x, 3);
    CHECK_DIM(weight, 2);
    CHECK_CPU(indices, 3);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(x.size(0) == indices.size(0));
    TORCH_CHECK(x.scalar_type() == weight.scalar_type());

    // sizes
    index_t batch_size = x.size(0);
    index_t in_blocks = x.size(1);
    index_t block_size = x.size(-1);
    index_t topk = indices.size(-1);
    index_t out_features = weight.size(0);
    TORCH_CHECK(out_features % block_size == 0);
    index_t out_blocks = out_features / block_size;
    // auto output = torch::zeros({batch_size, out_features}, x.options());

    //
    const auto indices_ptr = indices.accessor<int64_t, 3>();
    std::vector<std::vector<std::vector<index_t>>> groups(out_blocks);
    for (auto i = 0; i < out_blocks; i += 1) {
        groups[i].resize(in_blocks);
    }
    for (auto b = 0; b < batch_size; b += 1) {
        for (auto col = 0; col < in_blocks; col += 1) {
            for (auto i = 0; i < topk; i += 1) {
                auto row = indices_ptr[b][col][i];
                groups[row][col].push_back(b);
            }
        }
    }

    //
    std::vector<torch::Tensor> out_list;
    for (auto row = 0; row < out_blocks; row += 1) {
        auto y_i = torch::zeros(
            {batch_size, block_size}, x.options()
        );
        for (auto col = 0; col < in_blocks; col += 1) {
            if (groups[row][col].size() <= 0) {
                continue;
            }
            auto w_i = weight.index(
                {torch::indexing::Slice(
                     row * block_size, (row + 1) * block_size
                 ),
                 torch::indexing::Slice(
                     col * block_size, (col + 1) * block_size
                 )}
            );
            auto ids = torch::tensor(groups[row][col]).to(x.device());
            auto x_i = x.index({ids, col});
            torch::index_add_out(y_i, y_i, 0, ids, torch::matmul(x_i, w_i.t()));
        }
        out_list.push_back(y_i);
    }
    auto output = torch::cat(out_list, -1);

    //
    return output;
}