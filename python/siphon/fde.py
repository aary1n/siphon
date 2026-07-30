"""
Helpers for the C++ 2D finite-difference eigenmode (FDE) solver.

Builds refractive-index grids for rectangular strip waveguides and wraps
`siphon.solver.solve_mode` with sensible defaults. The heavy numerics live
in the C++ extension (`siphon.solver`, built via CMake); this module only
prepares inputs and interprets outputs.

Grid convention:
    Arrays have shape (ny, nx): row j is the y (height) index, column i is
    the x (width) index. Dirichlet walls (E = 0) sit half a cell outside
    the outermost grid points, so a domain of physical size Lx contains
    nx = round(Lx / dx) - 1 interior points.

Assumptions:
    - Staircase index sampling at grid points (no interface averaging).
      This limits interface accuracy to O(h); smooth profiles retain the
      stencil's O(h^2). Semi-vectorial interface treatment is a post-1.0
      stretch goal (see ROADMAP.md).
    - Scalar approximation: best suited to TE-like modes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from siphon.sensitivity import N_OXIDE, N_SILICON

try:
    from siphon import solver as _solver
except ImportError as exc:  # pragma: no cover - depends on local build
    _solver = None
    _IMPORT_ERROR = exc


def _require_solver():
    if _solver is None:
        raise ImportError(
            "The C++ extension siphon.solver is not built. Run the CMake "
            "build (see CLAUDE.md 'Build & Run') to produce "
            "python/siphon/solver*.pyd"
        ) from _IMPORT_ERROR
    return _solver


def build_rect_waveguide_n(
    width: float,
    height: float,
    domain_x: float,
    domain_y: float,
    dx: float,
    dy: float,
    n_core: float = N_SILICON,
    n_clad: float = N_OXIDE,
) -> NDArray[np.float64]:
    """
    Build the refractive-index grid n(y, x) for a centered rectangular core.

    Parameters
    ----------
    width, height : float
        Core dimensions [m].
    domain_x, domain_y : float
        Physical domain size [m]. Must leave enough cladding around the
        core for the field to decay before the Dirichlet walls
        (>= 600nm padding is typical at 1550nm).
    dx, dy : float
        Grid spacings [m].
    n_core, n_clad : float
        Core / cladding refractive indices.

    Returns
    -------
    n_grid : ndarray, shape (ny, nx), float64, C-contiguous
        Refractive index profile suitable for `siphon.solver.solve_mode`.
    """
    if domain_x < width or domain_y < height:
        raise ValueError("Domain must be larger than the waveguide core")

    nx = int(round(domain_x / dx)) - 1
    ny = int(round(domain_y / dy)) - 1
    if nx < 3 or ny < 3:
        raise ValueError("Grid too coarse: fewer than 3 interior points")

    # Interior points; Dirichlet walls half a cell outside the first/last.
    x = (np.arange(nx) + 1) * dx - domain_x / 2
    y = (np.arange(ny) + 1) * dy - domain_y / 2
    X, Y = np.meshgrid(x, y)

    in_core = (np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2)
    return np.where(in_core, n_core, n_clad).astype(np.float64)


def solve_waveguide_mode(
    width: float,
    height: float,
    wavelength: float = 1.55e-6,
    dx: float = 10e-9,
    dy: float = 10e-9,
    pad: float = 700e-9,
    n_core: float = N_SILICON,
    n_clad: float = N_OXIDE,
    n_guess: float = -1.0,
):
    """
    Solve the fundamental mode of a rectangular strip waveguide.

    Convenience wrapper: builds the index grid with `pad` of cladding on
    every side and calls the C++ solver.

    Returns
    -------
    result : siphon.solver.ModeResult
        n_eff, field (ny, nx), iterations.
    """
    s = _require_solver()
    n_grid = build_rect_waveguide_n(
        width, height,
        domain_x=width + 2 * pad,
        domain_y=height + 2 * pad,
        dx=dx, dy=dy, n_core=n_core, n_clad=n_clad,
    )
    return s.solve_mode(n_grid, wavelength, dx, dy, n_guess=n_guess)
