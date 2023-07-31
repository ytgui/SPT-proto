// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

torch::Tensor sparse_mha_forward(
    const torch::Tensor &query, const torch::Tensor &key
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mha_forward", &sparse_mha_forward, "sparse MHA forward");
}
