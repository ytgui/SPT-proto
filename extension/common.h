#ifndef COMMON_HEADER_FILE_H
#define COMMON_HEADER_FILE_H

// clang-format off
#include <vector>
#include <torch/extension.h>
// clang-format on

#define CHECK_DIM(x, d)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")                 \
    TORCH_CHECK(x.dim() == d, #x " must be of dim " #d);                  \
    TORCH_CHECK(                                                          \
        x.is_contiguous(), #x " custom kernel requires contiguous tensor" \
    )

#define CHECK_TYPE(x, t) \
    TORCH_CHECK(x.scalar_type() == t, #x " must be type of " #t)

using index_t = int64_t;

#endif
