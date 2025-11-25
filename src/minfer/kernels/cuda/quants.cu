#include <torch/extension.h>

// TODO: complete me!
void _dequant_row() {
    TORCH_CHECK(false, "_dequant_row not implemented");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_dequant_row", &_dequant_row);
}