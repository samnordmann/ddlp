#pragma once
#include <torch/torch.h>

namespace ddlp {
namespace primitives {

class LinearColumnwiseImpl {
public:
    LinearColumnwiseImpl(int64_t in_features, int64_t out_features);
    torch::Tensor forward(
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias);

private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t local_out_features_;
    int world_size_;
    void ensure_mpi_initialized();
};

} // namespace primitives
} // namespace ddlp

