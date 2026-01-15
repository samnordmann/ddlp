#include "ddlp/primitives/linear_columnwise.hpp"
#include <mpi.h>
#include <stdexcept>

namespace ddlp {
namespace primitives {

namespace {

MPI_Datatype mpi_dtype_for_scalar(at::ScalarType scalar_type) {
    switch (scalar_type) {
        case at::kFloat:
            return MPI_FLOAT;
        case at::kDouble:
            return MPI_DOUBLE;
        case at::kHalf:
        case at::kBFloat16:
            return MPI_UINT16_T;
        default:
            throw std::runtime_error("LinearColumnwise: unsupported dtype for MPI allgather");
    }
}

} // namespace

LinearColumnwiseImpl::LinearColumnwiseImpl(int64_t in_features, int64_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      local_out_features_(0),
      world_size_(1) {
    ensure_mpi_initialized();
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    if (out_features_ % world_size_ != 0) {
        throw std::runtime_error("LinearColumnwise: out_features must be divisible by world_size");
    }
    local_out_features_ = out_features_ / world_size_;
}

void LinearColumnwiseImpl::ensure_mpi_initialized() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }
}

torch::Tensor LinearColumnwiseImpl::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {
    if (!input.device().is_cpu() || !weight.device().is_cpu() ||
        (bias.has_value() && !bias->device().is_cpu())) {
        throw std::runtime_error("LinearColumnwise: only CPU tensors are supported");
    }
    if (input.scalar_type() != weight.scalar_type() ||
        (bias.has_value() && bias->scalar_type() != weight.scalar_type())) {
        throw std::runtime_error("LinearColumnwise: input/weight/bias dtypes must match");
    }
    if (input.size(-1) != in_features_) {
        throw std::runtime_error("LinearColumnwise: input last dim mismatch");
    }
    if (weight.dim() != 2 || weight.size(1) != in_features_) {
        throw std::runtime_error("LinearColumnwise: weight shape must be [local_out, in_features]");
    }
    if (weight.size(0) != local_out_features_) {
        throw std::runtime_error("LinearColumnwise: weight local_out_features mismatch");
    }
    if (bias.has_value() && bias->numel() != local_out_features_) {
        throw std::runtime_error("LinearColumnwise: bias shape mismatch");
    }

    auto weight_contig = weight.contiguous();
    auto full_weight = torch::empty({out_features_, in_features_}, weight_contig.options());
    auto full_weight_contig = full_weight.contiguous();

    const auto weight_numel = weight_contig.numel();
    const auto full_weight_numel = full_weight_contig.numel();
    if (full_weight_numel != weight_numel * world_size_) {
        throw std::runtime_error("LinearColumnwise: weight size mismatch for allgather");
    }

    MPI_Datatype dtype = mpi_dtype_for_scalar(weight_contig.scalar_type());
    MPI_Allgather(
        weight_contig.data_ptr(),
        static_cast<int>(weight_numel),
        dtype,
        full_weight_contig.data_ptr(),
        static_cast<int>(weight_numel),
        dtype,
        MPI_COMM_WORLD);

    c10::optional<torch::Tensor> full_bias = c10::nullopt;
    if (bias.has_value()) {
        auto bias_contig = bias->contiguous();
        auto full_bias_tensor = torch::empty({out_features_}, bias_contig.options());
        auto full_bias_contig = full_bias_tensor.contiguous();

        const auto bias_numel = bias_contig.numel();
        const auto full_bias_numel = full_bias_contig.numel();
        if (full_bias_numel != bias_numel * world_size_) {
            throw std::runtime_error("LinearColumnwise: bias size mismatch for allgather");
        }

        MPI_Allgather(
            bias_contig.data_ptr(),
            static_cast<int>(bias_numel),
            dtype,
            full_bias_contig.data_ptr(),
            static_cast<int>(bias_numel),
            dtype,
            MPI_COMM_WORLD);

        full_bias = full_bias_contig;
    }

    auto output = torch::matmul(input, full_weight_contig.t());
    if (full_bias.has_value()) {
        output = output + full_bias.value();
    }
    return output;
}

} // namespace primitives
} // namespace ddlp

