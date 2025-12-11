#include <torch/extension.h>
#include "ddlp/bindings.hpp"

PYBIND11_MODULE(_C, m) {
    m.doc() = "DDLP: Distributed Deep Learning Primitives C++ Backend";

    ddlp::bind_primitives(m);
    ddlp::bind_communicator(m);
}



