# Notes:
# In newer CMake, we can use CMAKE_CUDA_FLAGS 
# or target_compile_options(<target> PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:..>)
# It's usually better to attach flags to *targets* (target_compile_options)
# Instead of include_directories(), we can use target_include_directories()
# Missing: Targets

# Project setup
cmake_minimum_required(VERSION 3.26)
project(ada-fp8-gemm LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Compiler flags
# -O3 := optimization level (max)
# -fPIC := position-independent code
# -Wno-psabi := disables ABI warning messages (for PyTorch)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wno-psabi")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi")

# NVCC Compiler flags
# -DENABLE_FAST_DEBUG := preprocessor definition
# -O3 := optimization level (max)
# --ptrax-options := passes options to the PTX assembler:
#   --verbose :=  prints register usage, spills, etc.
#   --register-usage-level=10 := register allocation verbosity
#   --warn-on-local-memory-usage := warn on spill to local memory
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose, --register-usage-level=10, --warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "8.9" CACHE STRING "Which CUDA architecture to compile for")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

# Nanobind setup
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

include_directories(third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# Add a Python API entrance
# target_link_libraries...

# Add targets..

# Building our extension
# nanobind_add_module(my_ext my_ext.cpp)
