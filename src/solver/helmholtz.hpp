// SiPhON - 2D scalar Helmholtz mode solver.
//
// Solves the scalar eigenvalue problem for guided modes of a dielectric
// waveguide cross-section:
//
//     [ d2/dx2 + d2/dy2 + k0^2 n^2(x, y) ] E(x, y) = beta^2 E(x, y)
//
// discretized with a 5-point finite-difference stencil on a uniform grid
// with Dirichlet (E = 0) boundary conditions. The effective index of a
// mode is n_eff = beta / k0.
//
// Assumptions (documented per project policy):
//   - Scalar approximation: no polarization coupling, continuous field and
//     derivative across dielectric interfaces (semi-vectorial corrections
//     are a post-1.0 stretch goal). Best suited to TE-like modes.
//   - Dirichlet boundaries: the domain must be padded with enough cladding
//     that the guided field has decayed at the walls.
//   - The operator is real symmetric; the fundamental mode is the
//     algebraically largest eigenvalue beta^2.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "grid.hpp"

namespace siphon {

// Row-major matrix type matching NumPy's default (C-order) layout, so
// pybind11 can map arrays into Eigen::Ref without copying.
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct ModeResult {
    double n_eff = 0.0;         // Effective index beta / k0
    RowMajorMatrixXd field;     // Mode profile, shape (ny, nx), unit L2 norm
    int iterations = 0;         // Arnoldi restarts used by the eigensolver
};

class HelmholtzSolver {
public:
    explicit HelmholtzSolver(const Grid2D& grid) : grid_(grid) {}

    // Assemble the sparse symmetric operator
    //     H = L + k0^2 diag(n^2)
    // where L is the 5-point Dirichlet Laplacian. `n_squared` has shape
    // (ny, nx): row j is the y index, column i is the x index.
    Eigen::SparseMatrix<double> AssembleOperator(
        const Eigen::Ref<const RowMajorMatrixXd>& n_squared, double k0) const;

    // Find the fundamental mode: the eigenvalue beta^2 closest to
    // sigma = (k0 * n_guess)^2 via shift-and-invert Arnoldi iteration.
    //
    // If n_guess <= 0 it defaults to max(n) * (1 + 1e-4), which places the
    // shift strictly above the whole spectrum (the Laplacian is negative
    // semi-definite, so beta^2 < k0^2 max(n^2)); the closest eigenvalue is
    // then guaranteed to be the fundamental mode.
    ModeResult SolveFundamental(
        const Eigen::Ref<const RowMajorMatrixXd>& n_squared,
        double k0,
        double n_guess = -1.0,
        int max_iter = 1000,
        double tol = 1e-10,
        int ncv = 20) const;

    const Grid2D& grid() const { return grid_; }

private:
    Grid2D grid_;
};

}  // namespace siphon
