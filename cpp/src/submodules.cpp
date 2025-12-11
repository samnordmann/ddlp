#include "ddlp/bindings.hpp"
#include "ddlp/primitives/row_parallel.hpp"
#include "ddlp/primitives/col_parallel.hpp"
#include "ddlp/primitives/moe.hpp"
#include "ddlp/communicator.hpp"

namespace ddlp {

void bind_primitives(py::module_& m) {
    auto prim = m.def_submodule("primitives", "DDLP Primitives");
    
    using namespace primitives;
    
    py::class_<SPTPRowParallelLinearImpl>(prim, "SPTPRowParallelLinearImpl")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &SPTPRowParallelLinearImpl::forward);

    py::class_<SPTPColumnParallelLinearImpl>(prim, "SPTPColumnParallelLinearImpl")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &SPTPColumnParallelLinearImpl::forward);

    py::class_<TPRowParallelLinearImpl>(prim, "TPRowParallelLinearImpl")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &TPRowParallelLinearImpl::forward);

    py::class_<MixtureOfExpertsImpl>(prim, "MixtureOfExpertsImpl")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &MixtureOfExpertsImpl::forward);
}

void bind_communicator(py::module_& m) {
    auto comm = m.def_submodule("communicator", "DDLP Communicator");
    
    py::class_<CommunicatorImpl>(comm, "CommunicatorImpl")
        .def(py::init<>())
        .def("rank", &CommunicatorImpl::rank)
        .def("world_size", &CommunicatorImpl::world_size)
        .def("barrier", &CommunicatorImpl::barrier);
}

} // namespace ddlp
