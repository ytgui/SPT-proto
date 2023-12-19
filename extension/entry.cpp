// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

std::vector<torch::Tensor> cdist_forward_cuda(
    const torch::Tensor &query, const torch::Tensor &table
);

std::vector<torch::Tensor> cdist_backward_cuda(
    const torch::Tensor &query, const torch::Tensor &table,
    const torch::Tensor &grad_output
);

torch::Tensor lookup_forward_cuda(
    const torch::Tensor &config, const torch::Tensor &query,
    const torch::Tensor &key
);

torch::Tensor spmm_forward_cuda(
    const torch::Tensor &trans_lhs, const torch::Tensor &trans_rhs,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values, const torch::Tensor &x
);

torch::Tensor sddmm_forward_cuda(
    const torch::Tensor &trans_lhs, const torch::Tensor &trans_rhs,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &query, const torch::Tensor &key
);

torch::Tensor softmax_forward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &values
);

torch::Tensor softmax_backward_cuda(
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &output, const torch::Tensor &grad_output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // cdist
    m.def("cdist_forward_cuda", &cdist_forward_cuda, "cdist forward");
    m.def("cdist_backward_cuda", &cdist_backward_cuda, "cdist backward");
    // lookup
    m.def("lookup_forward_cuda", &lookup_forward_cuda, "lookup forward");
    // spmm
    m.def("spmm_forward_cuda", &spmm_forward_cuda, "spmm forward");
    // sddmm
    m.def("sddmm_forward_cuda", &sddmm_forward_cuda, "sddmm forward");
    // softmax
    m.def("softmax_forward_cuda", &softmax_forward_cuda, "softmax forward");
    m.def("softmax_backward_cuda", &softmax_backward_cuda, "softmax backward");
}
