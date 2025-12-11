#pragma once
#include <torch/torch.h>

namespace ddlp {
namespace primitives {

class SPTPColumnParallelLinearImpl {
public:
    SPTPColumnParallelLinearImpl(int64_t in_features, int64_t out_features);
    torch::Tensor forward(torch::Tensor input, torch::Tensor weight);
private:
    int64_t in_features_;
    int64_t out_features_;
};

} // namespace primitives
} // namespace ddlp



