#include <torch/extension.h>

// TODO: complete me!
void dequant_row() {
    TORCH_CHECK(false, "_dequant_row not implemented");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_row", &dequant_row);
}