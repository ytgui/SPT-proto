// clang-format off
#include <string>
#include <vector>
#include <torch/extension.h>
// clang-format on

#define CHECK_INPUT(x, d, t)                                                                                           \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                                                               \
    TORCH_CHECK(x.scalar_type() == t, #x " must be type of " #t);                                                      \
    TORCH_CHECK(x.is_contiguous(), #x " custom kernel requires contiguous tensor")

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#m.def("mha_forward", &mha_forward, "MHA forward");
}
