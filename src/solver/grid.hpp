// SiPhON - 2D spatial grid for the finite-difference Helmholtz solver.
//
// The grid covers the interior of the computational domain. Dirichlet
// boundary conditions (E = 0) are applied on the domain boundary, which
// lies half a cell outside the outermost grid points; boundary points are
// therefore not stored as unknowns.

#pragma once

#include <stdexcept>

namespace siphon {

// Uniform 2D grid of nx * ny interior points with spacings dx, dy [m].
//
// Index convention (matches a row-major NumPy array of shape (ny, nx)):
//   - i in [0, nx): x-direction (columns, width axis)
//   - j in [0, ny): y-direction (rows, height axis)
//   - linear index = j * nx + i
struct Grid2D {
    int nx;
    int ny;
    double dx;
    double dy;

    Grid2D(int nx_, int ny_, double dx_, double dy_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_) {
        if (nx < 3 || ny < 3) {
            throw std::invalid_argument("Grid2D: nx and ny must be >= 3");
        }
        if (dx <= 0.0 || dy <= 0.0) {
            throw std::invalid_argument("Grid2D: dx and dy must be positive");
        }
    }

    int size() const { return nx * ny; }

    int index(int i, int j) const { return j * nx + i; }
};

}  // namespace siphon
