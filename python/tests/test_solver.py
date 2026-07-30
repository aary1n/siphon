"""
Validation tests for the C++ 2D FDE Helmholtz solver (siphon.solver).

Verifies, in order of increasing physics:
1. Operator assembly matches an independent scipy.sparse construction
   exactly (same stencil, same index convention).
2. Homogeneous medium: the fundamental eigenvalue matches the closed-form
   eigenvalue of the discrete Dirichlet Laplacian to solver tolerance.
3. Slab reduction: a y-invariant... (x-invariant) slab profile reproduces
   the analytical slab waveguide n_eff after the exact separable
   x-confinement correction, to < 1% (Phase 0.3 acceptance criterion).
4. Full 2D strip waveguide: n_eff is physically bounded and consistent
   with the Effective Index Method estimate.
5. Grid convergence: observed order ~ O(h^2) for a smooth index profile
   (staircase interfaces would mask the stencil order).
6. Cross-check against scipy.sparse.linalg.eigsh on the same matrix.

The whole suite skips cleanly when the extension has not been built.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

solver = pytest.importorskip(
    "siphon.solver", reason="C++ extension not built (see CLAUDE.md Build & Run)"
)

from siphon.fde import build_rect_waveguide_n, solve_waveguide_mode
from siphon.sensitivity import (
    EffectiveIndexSolver,
    WaveguideGeometry,
    N_OXIDE,
    N_SILICON,
    slab_te_neff,
)

WAVELENGTH = 1.55e-6
K0 = 2 * np.pi / WAVELENGTH


def dirichlet_laplacian_eigenvalue(n_points: int, spacing: float, mode: int = 1) -> float:
    """
    Exact eigenvalue of the (positive) 1D second-difference operator with
    Dirichlet ends: lambda_m = (4 / d^2) * sin^2(m*pi / (2*(N+1))).
    """
    return 4.0 / spacing**2 * np.sin(mode * np.pi / (2 * (n_points + 1))) ** 2


# ---------------------------------------------------------------------------
# 1. Operator assembly
# ---------------------------------------------------------------------------

class TestOperatorAssembly:
    def test_matches_scipy_reference(self) -> None:
        """C++ assembly must equal an independent scipy construction exactly."""
        sp = pytest.importorskip("scipy.sparse")

        nx, ny = 13, 9
        dx, dy = 15e-9, 25e-9
        rng = np.random.default_rng(0)
        n_sq = rng.uniform(1.0, 12.0, size=(ny, nx))

        grid = solver.Grid2D(nx, ny, dx, dy)
        H_cpp = solver.HelmholtzSolver(grid).assemble_operator(n_sq, K0)

        N = nx * ny
        main = (-2 / dx**2 - 2 / dy**2) + K0**2 * n_sq.ravel()
        ex = np.ones(N - 1)
        ex[nx - 1 :: nx] = 0  # no coupling across row ends
        ey = np.ones(N - nx)
        H_ref = (
            sp.diags(main)
            + sp.diags(ex / dx**2, 1)
            + sp.diags(ex / dx**2, -1)
            + sp.diags(ey / dy**2, nx)
            + sp.diags(ey / dy**2, -nx)
        )

        assert abs(H_cpp - H_ref.tocsc()).max() == 0.0

    def test_shape_mismatch_raises(self) -> None:
        grid = solver.Grid2D(10, 8, 10e-9, 10e-9)
        with pytest.raises(Exception):
            solver.HelmholtzSolver(grid).assemble_operator(np.ones((5, 5)), K0)


# ---------------------------------------------------------------------------
# 2. Homogeneous medium (exact discrete solution)
# ---------------------------------------------------------------------------

class TestHomogeneousMedium:
    def test_exact_eigenvalue(self) -> None:
        """beta^2 = k0^2 n^2 - lambda_x1 - lambda_y1 exactly (discrete)."""
        nx, ny = 41, 31
        dx = dy = 20e-9
        n0 = 2.0

        result = solver.solve_mode(np.full((ny, nx), n0), WAVELENGTH, dx, dy)

        beta_sq_exact = (
            K0**2 * n0**2
            - dirichlet_laplacian_eigenvalue(nx, dx)
            - dirichlet_laplacian_eigenvalue(ny, dy)
        )
        n_eff_exact = np.sqrt(beta_sq_exact) / K0

        assert result.n_eff == pytest.approx(n_eff_exact, rel=1e-10)

    def test_fundamental_mode_shape(self) -> None:
        """Fundamental mode of a box: single-lobed, positive, unit norm."""
        nx, ny = 41, 31
        result = solver.solve_mode(
            np.full((ny, nx), 2.0), WAVELENGTH, 20e-9, 20e-9
        )

        field = result.field
        assert field.shape == (ny, nx)
        assert np.linalg.norm(field) == pytest.approx(1.0, rel=1e-9)
        # Sign convention: maximum is positive; fundamental has no nodes.
        assert field.max() > 0
        assert field.min() > -1e-8
        # Peak in the interior (analytically at the domain center)
        j, i = np.unravel_index(np.argmax(field), field.shape)
        assert abs(i - (nx - 1) / 2) <= 1
        assert abs(j - (ny - 1) / 2) <= 1

    def test_shift_invariance(self) -> None:
        """Result must not depend on the shift target n_guess."""
        n_grid = np.full((31, 41), 2.0)
        r_auto = solver.solve_mode(n_grid, WAVELENGTH, 20e-9, 20e-9)
        r_near = solver.solve_mode(n_grid, WAVELENGTH, 20e-9, 20e-9, n_guess=1.35)
        assert r_auto.n_eff == pytest.approx(r_near.n_eff, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Slab reduction (analytical limit, Phase 0.3 acceptance: < 1%)
# ---------------------------------------------------------------------------

class TestSlabReduction:
    def test_slab_within_1_percent(self) -> None:
        """
        x-invariant slab: the 2D problem separates as
            n_eff_2D^2 = n_eff_slab^2 - lambda_x1 / k0^2
        with lambda_x1 the exact discrete x-confinement eigenvalue. After
        adding the correction back, the result must match the analytical
        (continuous) slab TE solution to < 1%.
        """
        thickness = 220e-9
        dy = 5e-9
        dx = 20e-9
        domain_y = 1.6e-6
        domain_x = 4.0e-6

        nx = int(round(domain_x / dx)) - 1
        ny = int(round(domain_y / dy)) - 1

        y = (np.arange(ny) + 1) * dy - domain_y / 2
        n_row = np.where(np.abs(y) <= thickness / 2, N_SILICON, N_OXIDE)
        n_grid = np.tile(n_row[:, None], (1, nx))

        result = solver.solve_mode(n_grid, WAVELENGTH, dx, dy)

        lambda_x1 = dirichlet_laplacian_eigenvalue(nx, dx)
        n_eff_slab_discrete = np.sqrt(result.n_eff**2 + lambda_x1 / K0**2)

        n_eff_analytical = slab_te_neff(thickness, WAVELENGTH, N_SILICON, N_OXIDE)

        rel_error = abs(n_eff_slab_discrete - n_eff_analytical) / n_eff_analytical
        assert rel_error < 0.01, (
            f"Slab n_eff {n_eff_slab_discrete:.4f} vs analytical "
            f"{n_eff_analytical:.4f}: {rel_error:.2%}"
        )


# ---------------------------------------------------------------------------
# 4. Full 2D strip waveguide
# ---------------------------------------------------------------------------

class TestStripWaveguide:
    @pytest.fixture(scope="class")
    def strip_result(self):
        return solve_waveguide_mode(500e-9, 220e-9, dx=10e-9, dy=10e-9)

    def test_neff_bounded(self, strip_result) -> None:
        assert N_OXIDE < strip_result.n_eff < N_SILICON

    def test_consistent_with_te_te_eim(self, strip_result) -> None:
        """
        The scalar Helmholtz approximation enforces field continuity across
        all interfaces, i.e. it misses the Ex discontinuity at the
        sidewalls -- the same physics the TE-TE variant of the EIM misses.
        The apples-to-apples check is therefore scalar FDE vs TE-TE EIM
        (~1% expected). The TM-corrected EIM (siphon.sensitivity) sits
        ~6% below both; that systematic scalar error is a documented
        limitation (see ADR-001 / ROADMAP semi-vectorial stretch goal).
        """
        n_slab = slab_te_neff(220e-9, WAVELENGTH, N_SILICON, N_OXIDE)
        eim_te_te = slab_te_neff(500e-9, WAVELENGTH, n_slab, N_OXIDE)
        assert strip_result.n_eff == pytest.approx(eim_te_te, rel=0.02)

    def test_scalar_error_vs_corrected_eim_documented(self, strip_result) -> None:
        """Scalar FDE sits above the TM-corrected EIM by a known ~5-8%."""
        eim = EffectiveIndexSolver(
            WaveguideGeometry(width=500e-9, height=220e-9)
        ).n_eff()
        rel = (strip_result.n_eff - eim) / eim
        assert 0.0 < rel < 0.10, f"Scalar-vs-EIM offset {rel:.2%} outside expected band"

    def test_field_confined_to_core(self, strip_result) -> None:
        """Field energy concentrated near the core; walls essentially zero."""
        field = strip_result.field
        ny, nx = field.shape
        # Peak within the core region (center of the domain)
        j, i = np.unravel_index(np.argmax(np.abs(field)), field.shape)
        assert abs(i - nx / 2) < nx / 6
        assert abs(j - ny / 2) < ny / 6
        # Dirichlet padding sufficient: boundary values negligible.
        # 700nm padding leaves ~1e-4 relative field at the walls; the
        # induced n_eff error scales as field^2 (~1e-8), far below the
        # discretization error.
        edge_max = max(
            np.abs(field[0, :]).max(), np.abs(field[-1, :]).max(),
            np.abs(field[:, 0]).max(), np.abs(field[:, -1]).max(),
        )
        assert edge_max < 1e-3 * np.abs(field).max()

    def test_neff_increases_with_width(self) -> None:
        r_narrow = solve_waveguide_mode(420e-9, 220e-9, dx=10e-9, dy=10e-9)
        r_wide = solve_waveguide_mode(580e-9, 220e-9, dx=10e-9, dy=10e-9)
        assert r_wide.n_eff > r_narrow.n_eff


# ---------------------------------------------------------------------------
# 5. Grid convergence (O(h^2) for smooth profiles)
# ---------------------------------------------------------------------------

class TestGridConvergence:
    def test_second_order_convergence_smooth_profile(self) -> None:
        """
        Richardson estimate of the convergence order on a smooth (Gaussian)
        graded-index profile:
            p = log2( (f_4h - f_2h) / (f_2h - f_h) )  ->  2 for the 5-point
        stencil. A smooth profile is used deliberately: staircase interface
        sampling is O(h) and would mask the stencil order.

        Each h must divide the domain exactly so the refinement levels
        share the same walls and nest (otherwise the domain itself changes
        with h and pollutes the Richardson estimate).
        """
        domain = 1.0e-6
        sigma = 0.15e-6

        def solve_at(h: float) -> float:
            n_cells = domain / h
            assert abs(n_cells - round(n_cells)) < 1e-9, "h must divide domain"
            n = int(round(n_cells)) - 1
            c = (np.arange(n) + 1) * h - domain / 2
            X, Y = np.meshgrid(c, c)
            n_grid = 1.5 + 1.5 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            return solver.solve_mode(n_grid, WAVELENGTH, h, h).n_eff

        f_4h = solve_at(20e-9)
        f_2h = solve_at(10e-9)
        f_h = solve_at(5e-9)

        order = np.log2(abs(f_4h - f_2h) / abs(f_2h - f_h))
        assert 1.7 < order < 2.3, f"Observed convergence order {order:.2f}"


# ---------------------------------------------------------------------------
# 6. Cross-check against scipy (same matrix, independent eigensolver)
# ---------------------------------------------------------------------------

class TestScipyCrossCheck:
    def test_waveguide_eigenvalue_matches_eigsh(self) -> None:
        spla = pytest.importorskip("scipy.sparse.linalg")

        dx = dy = 20e-9
        n_grid = build_rect_waveguide_n(
            500e-9, 220e-9, domain_x=1.7e-6, domain_y=1.4e-6, dx=dx, dy=dy
        )
        ny, nx = n_grid.shape

        r = solver.solve_mode(n_grid, WAVELENGTH, dx, dy)

        grid = solver.Grid2D(nx, ny, dx, dy)
        H = solver.HelmholtzSolver(grid).assemble_operator(n_grid**2, K0)
        sigma = (K0 * N_SILICON * 1.0001) ** 2
        beta_sq = spla.eigsh(
            H.tocsc(), k=1, sigma=sigma, which="LM", return_eigenvectors=False
        )[0]

        assert r.n_eff == pytest.approx(np.sqrt(beta_sq) / K0, rel=1e-9)
