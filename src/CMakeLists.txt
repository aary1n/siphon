# Core numerics as a static library (reused by future C++ tests) plus the
# pybind11 extension module.

add_library(siphon_core STATIC
    solver/helmholtz.cpp
)
target_include_directories(siphon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(siphon_core PUBLIC eigen_headers spectra_headers)

pybind11_add_module(solver bindings/pybind_solver.cpp)
target_link_libraries(solver PRIVATE siphon_core)

# Place the extension next to the pure-Python package so `import
# siphon.solver` works from the source tree (per CLAUDE.md layout).
set_target_properties(solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/siphon
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/siphon
)
