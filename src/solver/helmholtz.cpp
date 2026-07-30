#include "helmholtz.hpp"

#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace siphon {

Eigen::SparseMatrix<double> HelmholtzSolver::AssembleOperator(
    const Eigen::Ref<const RowMajorMatrixXd>& n_squared, double k0) const {
    const int nx = grid_.nx;
    const int ny = grid_.ny;

    if (n_squared.rows() != ny || n_squared.cols() != nx) {
        throw std::invalid_argument(
            "AssembleOperator: n_squared shape (" +
            std::to_string(n_squared.rows()) + ", " +
            std::to_string(n_squared.cols()) + ") does not match grid (ny=" +
            std::to_string(ny) + ", nx=" + std::to_string(nx) + ")");
    }
    if (k0 <= 0.0) {
        throw std::invalid_argument("AssembleOperator: k0 must be positive");
    }

    const double inv_dx2 = 1.0 / (grid_.dx * grid_.dx);
    const double inv_dy2 = 1.0 / (grid_.dy * grid_.dy);
    const double k0_sq = k0 * k0;

    // 5-point stencil: at most 5 entries per row.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(grid_.size()) * 5);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int row = grid_.index(i, j);

            // Diagonal: Laplacian center + refractive index term.
            // Dirichlet BC: missing neighbors contribute zero field, so the
            // stencil coefficients are simply omitted (matrix stays symmetric).
            triplets.emplace_back(
                row, row, -2.0 * inv_dx2 - 2.0 * inv_dy2 + k0_sq * n_squared(j, i));

            if (i > 0) {
                triplets.emplace_back(row, grid_.index(i - 1, j), inv_dx2);
            }
            if (i < nx - 1) {
                triplets.emplace_back(row, grid_.index(i + 1, j), inv_dx2);
            }
            if (j > 0) {
                triplets.emplace_back(row, grid_.index(i, j - 1), inv_dy2);
            }
            if (j < ny - 1) {
                triplets.emplace_back(row, grid_.index(i, j + 1), inv_dy2);
            }
        }
    }

    Eigen::SparseMatrix<double> H(grid_.size(), grid_.size());
    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

ModeResult HelmholtzSolver::SolveFundamental(
    const Eigen::Ref<const RowMajorMatrixXd>& n_squared,
    double k0,
    double n_guess,
    int max_iter,
    double tol,
    int ncv) const {
    const int n = grid_.size();

    // Nondimensionalize: solve H' = H / k0^2, whose eigenvalues are
    // lambda' = (beta / k0)^2 = n_eff^2 ~ O(1). In SI units the
    // shift-inverted eigenvalues nu = 1 / (beta^2 - sigma) are ~1e-14 --
    // only tens of machine epsilons -- and the Arnoldi iteration falsely
    // converges (absolute-scale thresholds inside the eigensolver).
    // Scaling makes the arithmetic well-conditioned without changing
    // eigenvectors. Verified against scipy.sparse.linalg.eigsh on the
    // identical matrix.
    Eigen::SparseMatrix<double> H = AssembleOperator(n_squared, k0);
    H *= 1.0 / (k0 * k0);

    // Default shift: strictly above the top of the spectrum. The Laplacian
    // part is negative semi-definite, so all eigenvalues satisfy
    // n_eff^2 < max(n^2); the eigenvalue closest to the shift is then
    // the fundamental (largest n_eff^2) mode.
    if (n_guess <= 0.0) {
        const double n_max = std::sqrt(n_squared.maxCoeff());
        n_guess = n_max * (1.0 + 1e-4);
    }
    const double sigma = n_guess * n_guess;

    ncv = std::min(std::max(ncv, 3), n);

    Spectra::SparseSymShiftSolve<double> op(H);
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(
        op, 1, ncv, sigma);

    eigs.init();
    // In shift-and-invert mode the selection rule applies to
    // nu = 1 / (lambda' - sigma): LargestMagn selects the eigenvalue of H'
    // closest to sigma. Spectra maps nu back to lambda' internally.
    const int nconv = eigs.compute(Spectra::SortRule::LargestMagn, max_iter, tol);

    if (nconv < 1 || eigs.info() != Spectra::CompInfo::Successful) {
        throw std::runtime_error(
            "SolveFundamental: eigensolver failed to converge (nconv=" +
            std::to_string(nconv) + ", max_iter=" + std::to_string(max_iter) +
            "). Try a larger ncv or a different n_guess.");
    }

    const double n_eff_sq = eigs.eigenvalues()(0);
    if (n_eff_sq <= 0.0) {
        throw std::runtime_error(
            "SolveFundamental: converged eigenvalue n_eff^2 = " +
            std::to_string(n_eff_sq) +
            " is not a guided mode (must be positive). The domain may "
            "be too small or the structure below cutoff.");
    }

    ModeResult result;
    result.n_eff = std::sqrt(n_eff_sq);
    result.iterations = static_cast<int>(eigs.num_iterations());

    // Reshape the eigenvector (linear index j * nx + i) into (ny, nx).
    // Fix the overall sign so the field maximum is positive.
    Eigen::VectorXd v = eigs.eigenvectors().col(0);
    Eigen::Index max_loc;
    v.cwiseAbs().maxCoeff(&max_loc);
    if (v(max_loc) < 0.0) {
        v = -v;
    }
    result.field = Eigen::Map<const RowMajorMatrixXd>(v.data(), grid_.ny, grid_.nx);

    return result;
}

}  // namespace siphon
