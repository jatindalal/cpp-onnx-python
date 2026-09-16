#include <nanobind/nanobind.h>
#include <onnxruntime_cxx_api.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(cpp_onnx_python_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
}

