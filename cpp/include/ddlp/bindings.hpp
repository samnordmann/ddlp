#pragma once
#include <torch/extension.h>

namespace ddlp {

void bind_primitives(py::module_& m);
void bind_communicator(py::module_& m);

} // namespace ddlp



