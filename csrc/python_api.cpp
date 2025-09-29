#include <nanobind/nanobind.h>
#include <torch/python.h>

namespace nb = nanobind;

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME ada_fp8_gemm
#endif

// Temporary just for now
int add(int a, int b) { return a + b; }

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add);
}
