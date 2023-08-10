// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

torch::Tensor sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
);

std::vector<torch::Tensor> sparse_mha_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &output, const torch::Tensor &grad_output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mha_forward", &sparse_mha_forward_cuda, "sparse MHA forward");
    m.def("sparse_mha_backward", &sparse_mha_backward_cuda, "sparse MHA backward");
}
