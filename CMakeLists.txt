# The minimum required cmake version
cmake_minimum_required(VERSION 3.9)

# Add cmake modules of this project to the module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Name and details of the project
project(cusvna VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CUSVNA_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")

# Define build options
option(CUSVNA_BUILD_TESTS "Enable the compilation of the example files." ON)

# Find CUDA with C++17 support
find_package(CUDAToolkit 11 REQUIRED)
add_compile_definitions("AUTODIFF_CUDA_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIRS}")

# We need to rely on deprecated FindCUDA to determine device architecture
include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures or compute capability versions (6.1, 7.0, etc) to generate code for.")
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

# cusvna requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Build the tests
if(CUSVNA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
