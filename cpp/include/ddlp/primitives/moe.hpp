#pragma once
#include <torch/torch.h>

namespace ddlp {
namespace primitives {

class MixtureOfExpertsImpl {
public:
    MixtureOfExpertsImpl(int64_t num_experts, int64_t hidden_size);
    torch::Tensor forward(torch::Tensor input, torch::Tensor gate);
private:
    int64_t num_experts_;
    int64_t hidden_size_;
};

} // namespace primitives
} // namespace ddlp



