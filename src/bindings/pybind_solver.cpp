// SiPhON - pybind11 bindings for the C++ Helmholtz mode solver.
//
// Exposed as the `siphon.solver` module. Input refractive-index grids are
// mapped zero-copy from C-ordered (row-major) NumPy arrays into
// Eigen::Ref<const RowMajorMatrixXd>; no copy is made as long as the array
// is a contiguous float64 array.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "../solver/grid.hpp"
#include "../solver/helmholtz.hpp"

// MSVC does not define M_PI without _USE_MATH_DEFINES; use our own constant.
constexpr double kPi = 3.14159265358979323846;

namespace py = pybind11;
using siphon::Grid2D;
using siphon::HelmholtzSolver;
using siphon::ModeResult;
using siphon::RowMajorMatrixXd;

namespace {

// Convenience free function mirroring the CLAUDE.md API sketch:
// build the grid from the array shape, square the index profile, solve.
ModeResult solve_mode(
    const Eigen::Ref<const RowMajorMatrixXd>& n_grid,
    double wavelength,
    double dx,
    double dy,
    double n_guess,
    int max_iter,
    double tol,
    int ncv) {
    if (wavelength <= 0.0) {
        throw std::invalid_argument("solve_mode: wavelength must be positive");
    }
    const Grid2D grid(static_cast<int>(n_grid.cols()),
                      static_cast<int>(n_grid.rows()), dx, dy);
    const HelmholtzSolver solver(grid);
    const double k0 = 2.0 * kPi / wavelength;
    const RowMajorMatrixXd n_squared = n_grid.array().square();
    return solver.SolveFundamental(n_squared, k0, n_guess, max_iter, tol, ncv);
}

}  // namespace

PYBIND11_MODULE(solver, m) {
    m.doc() =
        "SiPhON C++ 2D scalar Helmholtz mode solver (5-point FD stencil, "
        "Spectra shift-and-invert Arnoldi eigensolver).";

    py::class_<ModeResult>(m, "ModeResult")
        .def_readonly("n_eff", &ModeResult::n_eff,
                      "Effective index of the fundamental mode (beta / k0).")
        .def_readonly("field", &ModeResult::field,
                      "Mode field profile, shape (ny, nx), unit L2 norm.")
        .def_readonly("iterations", &ModeResult::iterations,
                      "Arnoldi iterations used by the eigensolver.")
        .def("__repr__", [](const ModeResult& r) {
            return "ModeResult(n_eff=" + std::to_string(r.n_eff) +
                   ", iterations=" + std::to_string(r.iterations) + ")";
        });

    py::class_<Grid2D>(m, "Grid2D")
        .def(py::init<int, int, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"))
        .def_readonly("nx", &Grid2D::nx)
        .def_readonly("ny", &Grid2D::ny)
        .def_readonly("dx", &Grid2D::dx)
        .def_readonly("dy", &Grid2D::dy)
        .def_property_readonly("size", &Grid2D::size);

    py::class_<HelmholtzSolver>(m, "HelmholtzSolver")
        .def(py::init<const Grid2D&>(), py::arg("grid"))
        .def("assemble_operator", &HelmholtzSolver::AssembleOperator,
             py::arg("n_squared"), py::arg("k0"),
             "Assemble the sparse Helmholtz operator H = L + k0^2 diag(n^2). "
             "Returned as a scipy.sparse matrix (copied).")
        .def("solve_fundamental", &HelmholtzSolver::SolveFundamental,
             py::arg("n_squared"), py::arg("k0"), py::arg("n_guess") = -1.0,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-10,
             py::arg("ncv") = 20,
             py::call_guard<py::gil_scoped_release>(),
             "Solve for the fundamental mode of an n^2(y, x) profile.");

    m.def("solve_mode", &solve_mode,
          py::arg("n_grid"), py::arg("wavelength"), py::arg("dx"),
          py::arg("dy"), py::arg("n_guess") = -1.0, py::arg("max_iter") = 1000,
          py::arg("tol") = 1e-10, py::arg("ncv") = 20,
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Solve for the fundamental mode of a refractive-index cross-section.

Parameters
----------
n_grid : ndarray, shape (ny, nx), float64, C-contiguous
    Refractive index profile n(y, x) (not squared). Mapped zero-copy.
wavelength : float
    Vacuum wavelength [m].
dx, dy : float
    Grid spacings [m].
n_guess : float, optional
    Shift target for the eigensolver, sigma = (k0 * n_guess)^2.
    Default (-1): max(n) * (1 + 1e-4), which always brackets the
    fundamental mode from above.
max_iter, tol, ncv : optional
    Spectra Arnoldi parameters.

Returns
-------
ModeResult
    n_eff, field (ny, nx), iterations.
)doc");

    m.attr("__version__") = "0.3.0-dev";
}
