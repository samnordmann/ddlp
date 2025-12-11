#include "ddlp/primitives/col_parallel.hpp"
#include <iostream>

namespace ddlp {
namespace primitives {

SPTPColumnParallelLinearImpl::SPTPColumnParallelLinearImpl(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {}

torch::Tensor SPTPColumnParallelLinearImpl::forward(torch::Tensor input, torch::Tensor weight) {
    // Placeholder: In real implementation, this would invoke Fuser
    // Logic:
    // 1. AllGather(X)
    // 2. Matmul: Y = AG(X) * W
    
    // For scaffolding, just return a matmul
    return torch::matmul(input, weight.t());
}

} // namespace primitives
} // namespace ddlp



