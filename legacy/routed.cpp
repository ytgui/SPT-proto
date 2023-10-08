#include "common.h"

#define CHECK_CPU(x, d)                                                   \
    TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")                   \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

torch::Tensor routed_forward_cuda(
    const torch::Tensor &config, const torch::Tensor &indices,
    const torch::Tensor &weight, const torch::Tensor &x
) {
    CHECK_DIM(x, 2);
    CHECK_CPU(config, 1);
    CHECK_DIM(weight, 2);
    CHECK_CPU(indices, 2);
    CHECK_TYPE(indices, torch::kInt64);
    TORCH_CHECK(x.size(0) == indices.size(0));
    TORCH_CHECK(x.size(-1) == weight.size(-1));
    TORCH_CHECK(x.scalar_type() == weight.scalar_type());

    // sizes
    auto device = x.device();
    index_t batch_size = x.size(0);
    index_t in_features = x.size(-1);
    index_t n_experts = indices.size(-1);
    index_t out_features = weight.size(0);
    auto h = torch::zeros({batch_size, out_features}, x.options());

    // grouping
    std::vector<std::vector<index_t>> grouping(n_experts);
    const auto indices_ptr = indices.accessor<int64_t, 2>();
    for (auto b = 0; b < batch_size; b += 1) {
        for (auto i = 0; i < n_experts; i += 1) {
            auto expert = indices_ptr[b][i];
            grouping[expert].push_back(b);
        }
    }
    std::vector<index_t> indptr;
    std::vector<index_t> group_flat;
    for (auto i = 0; i < grouping.size(); i += 1) {
        indptr.push_back(grouping_flat.size());
        for (auto j = 0; j < grouping[i].size(); k += 1) {
            grouping_flat.push_back(grouping[i][j]);
        }
    }
    indptr.push_back(grouping_flat.size());
    auto group_tensor = torch::tensor(group_flat).to(device);

    // compute
    for (auto i = 0; i < indptr.size() - 1; i += 1) {
        index_t beg = indptr[i];
        index_t end = indptr[i + 1];
    }


    /*
        x_i = x[batches]
        h_i = h[batches, i]
        w_i = self.fc1.weight[
            i * self.block_size:(i + 1) * self.block_size
        ]
        h[batches, i] = torch.addmm(
            h_i, x_i, w_i.T, beta=1.0, alpha=1.0
        )
    */
}