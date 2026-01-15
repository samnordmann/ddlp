cmake_minimum_required(VERSION 3.18)
project(DDLP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set Torch Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Explicit Source Listing
set(DDLP_SOURCES
    cpp/src/bindings.cpp
    cpp/src/communicator.cpp
    cpp/src/submodules.cpp
    cpp/src/primitives/col_parallel.cpp
    cpp/src/primitives/linear_columnwise.cpp
    cpp/src/primitives/moe.cpp
    cpp/src/primitives/row_parallel.cpp
)

# Define the library
pybind11_add_module(_C ${DDLP_SOURCES})

# Include directories
target_include_directories(_C PUBLIC 
    cpp/include
    ${MPI_CXX_INCLUDE_DIRS}
)

# Find torch_python library for pybind11 ATen bindings
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "torch_python library not found, may have linking issues")
endif()

# Link against dependencies
target_link_libraries(_C PRIVATE 
    ${TORCH_LIBRARIES} 
    ${TORCH_PYTHON_LIBRARY} 
    CUDA::cudart 
    MPI::MPI_CXX
)

# Set RPATH handling
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Installation
install(TARGETS _C DESTINATION lib)
