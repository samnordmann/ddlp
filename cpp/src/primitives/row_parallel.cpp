#include "ddlp/primitives/row_parallel.hpp"
#include <iostream>

namespace ddlp {
namespace primitives {

SPTPRowParallelLinearImpl::SPTPRowParallelLinearImpl(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {}

torch::Tensor SPTPRowParallelLinearImpl::forward(torch::Tensor input, torch::Tensor weight) {
    // Placeholder: In real implementation, this would invoke Fuser or custom kernels
    // for Linear + ReduceScatter
    // Logic: 
    // 1. Matmul: Y = X * W
    // 2. ReduceScatter(Y)
    
    // For scaffolding, just return a matmul
    return torch::matmul(input, weight.t()); 
}

TPRowParallelLinearImpl::TPRowParallelLinearImpl(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {}

torch::Tensor TPRowParallelLinearImpl::forward(torch::Tensor input, torch::Tensor weight) {
    // Placeholder for Linear + AllReduce
    return torch::matmul(input, weight.t());
}

} // namespace primitives
} // namespace ddlp



