// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

torch::Tensor sddmm_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
);

torch::Tensor cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
);

std::vector<torch::Tensor> cdist_backward_cuda(
    const torch::Tensor &query, const torch::Tensor &table,
    const torch::Tensor &grad_output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sddmm_forward_cuda", &sddmm_forward_cuda, "SDDMM forward");
    m.def("cdist_forward_cuda", &cdist_forward_cuda, "cdist forward");
    m.def("cdist_backward_cuda", &cdist_backward_cuda, "cdist backward");
}
