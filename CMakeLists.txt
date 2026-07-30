# SiPhON - top-level build: C++ Helmholtz solver + pybind11 module.
#
# Dependencies are fetched at configure time and version-locked:
#   - Eigen 3.4.0   (header-only; its own CMakeLists is skipped, see below)
#   - Spectra 1.2.0 (header-only; its own CMakeLists is skipped)
#   - pybind11 3.0.4
#
# Eigen's and Spectra's CMakeLists predate CMake 4 minimum-version policy
# requirements, so we point SOURCE_SUBDIR at a directory that does not
# contain a CMakeLists.txt; FetchContent then downloads the sources without
# calling add_subdirectory(), and we consume the headers via INTERFACE
# targets instead.

cmake_minimum_required(VERSION 3.20)
project(siphon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_SUBDIR cmake_subdir_skipped
)
FetchContent_Declare(
    spectra
    URL https://github.com/yixuan/spectra/archive/refs/tags/v1.2.0.tar.gz
    SOURCE_SUBDIR cmake_subdir_skipped
)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.4.tar.gz
)
FetchContent_MakeAvailable(eigen spectra pybind11)

add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers INTERFACE ${eigen_SOURCE_DIR})

add_library(spectra_headers INTERFACE)
target_include_directories(spectra_headers INTERFACE ${spectra_SOURCE_DIR}/include)

add_subdirectory(src)
