// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

torch::Tensor matmul_cuda(
    const torch::Tensor &left, const torch::Tensor &right
);

torch::Tensor cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
);

std::vector<torch::Tensor> cdist_backward_cuda(
    const torch::Tensor &query, const torch::Tensor &table,
    const torch::Tensor &grad_output
);

std::vector<torch::Tensor> sparse_mha_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value
);

std::vector<torch::Tensor> sparse_mha_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, const torch::Tensor &attention,
    const torch::Tensor &output, const torch::Tensor &grad_output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "matmul forward");
    m.def("cdist_forward_cuda", &cdist_forward_cuda, "PQ cdist forward");
    m.def("cdist_backward_cuda", &cdist_backward_cuda, "PQ cdist backward");
    m.def("sparse_mha_forward", &sparse_mha_forward_cuda, "sparse MHA forward");
    m.def("sparse_mha_backward", &sparse_mha_backward_cuda, "sparse MHA backward");
}
