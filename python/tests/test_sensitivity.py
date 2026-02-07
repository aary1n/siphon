"""
Unit tests for the Effective Index Method sensitivity model.

Tests verify:
1. Slab waveguide solver produces valid n_eff in (n_clad, n_core)
2. EIM n_eff matches expected range for typical silicon wire (~2.35-2.45)
3. Sensitivities are positive and in expected magnitude range
4. n_eff increases with width and height (monotonicity)
5. Edge cases: near-cutoff, multimode waveguides
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siphon.sensitivity import (
    WaveguideGeometry,
    SensitivityCoefficients,
    EffectiveIndexSolver,
    slab_te_neff,
    check_single_mode,
    N_SILICON,
    N_OXIDE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def typical_waveguide() -> WaveguideGeometry:
    """Standard 500nm x 220nm SOI waveguide at 1550nm."""
    return WaveguideGeometry(width=500e-9, height=220e-9)


@pytest.fixture
def solver(typical_waveguide: WaveguideGeometry) -> EffectiveIndexSolver:
    """EIM solver at nominal geometry."""
    return EffectiveIndexSolver(typical_waveguide)


# ---------------------------------------------------------------------------
# TestWaveguideGeometry
# ---------------------------------------------------------------------------

class TestWaveguideGeometry:
    """Tests for WaveguideGeometry dataclass."""

    def test_valid_geometry(self) -> None:
        """Valid parameters should create geometry without error."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        assert wg.width == 500e-9
        assert wg.height == 220e-9
        assert wg.wavelength == 1.55e-6
        assert wg.n_core == N_SILICON
        assert wg.n_clad == N_OXIDE

    def test_invalid_width(self) -> None:
        """Non-positive width should raise ValueError."""
        with pytest.raises(ValueError, match="Width must be positive"):
            WaveguideGeometry(width=0, height=220e-9)
        with pytest.raises(ValueError, match="Width must be positive"):
            WaveguideGeometry(width=-100e-9, height=220e-9)

    def test_invalid_height(self) -> None:
        """Non-positive height should raise ValueError."""
        with pytest.raises(ValueError, match="Height must be positive"):
            WaveguideGeometry(width=500e-9, height=0)

    def test_invalid_indices(self) -> None:
        """Core index must exceed cladding index."""
        with pytest.raises(ValueError, match="Core index"):
            WaveguideGeometry(width=500e-9, height=220e-9, n_core=1.4, n_clad=1.5)
        with pytest.raises(ValueError, match="Core index"):
            WaveguideGeometry(width=500e-9, height=220e-9, n_core=1.5, n_clad=1.5)


# ---------------------------------------------------------------------------
# TestSlabSolver
# ---------------------------------------------------------------------------

class TestSlabSolver:
    """Tests for the 1D slab TE mode solver."""

    def test_neff_bounds(self) -> None:
        """n_eff must be between n_clad and n_core."""
        n_eff = slab_te_neff(
            thickness=220e-9, wavelength=1.55e-6,
            n_core=N_SILICON, n_clad=N_OXIDE,
        )
        assert N_OXIDE < n_eff < N_SILICON

    def test_neff_increases_with_thickness(self) -> None:
        """Thicker slab should have higher n_eff."""
        n_thin = slab_te_neff(150e-9, 1.55e-6, N_SILICON, N_OXIDE)
        n_thick = slab_te_neff(300e-9, 1.55e-6, N_SILICON, N_OXIDE)
        assert n_thick > n_thin

    def test_thick_slab_approaches_core(self) -> None:
        """Very thick slab n_eff should approach n_core."""
        n_eff = slab_te_neff(5000e-9, 1.55e-6, N_SILICON, N_OXIDE)
        assert n_eff > 0.95 * N_SILICON

    def test_known_220nm_slab(self) -> None:
        """220nm Si slab at 1550nm should give n_eff in known range."""
        n_eff = slab_te_neff(220e-9, 1.55e-6, N_SILICON, N_OXIDE)
        # For a 220nm Si slab, n_eff should be approximately 2.8-3.1
        assert 2.5 < n_eff < 3.3

    def test_very_thin_slab_raises(self) -> None:
        """Extremely thin slab should raise ValueError (below cutoff threshold)."""
        with pytest.raises(ValueError, match="too low|too thin|No guided mode"):
            slab_te_neff(10e-9, 1.55e-6, N_SILICON, N_OXIDE)


# ---------------------------------------------------------------------------
# TestEffectiveIndexMethod
# ---------------------------------------------------------------------------

class TestEffectiveIndexMethod:
    """Tests for the two-step EIM solver."""

    def test_typical_silicon_wire(self, solver: EffectiveIndexSolver) -> None:
        """500nm x 220nm Si wire at 1550nm: n_eff should be ~2.35-2.50."""
        n_eff = solver.n_eff()
        assert 2.2 < n_eff < 2.6, f"Got n_eff = {n_eff}"

    def test_neff_increases_with_width(self) -> None:
        """Wider waveguide should have higher n_eff."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        n_narrow = solver.n_eff(width=400e-9)
        n_wide = solver.n_eff(width=600e-9)
        assert n_wide > n_narrow

    def test_neff_increases_with_height(self) -> None:
        """Thicker silicon should give higher n_eff."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        n_thin = solver.n_eff(height=200e-9)
        n_thick = solver.n_eff(height=250e-9)
        assert n_thick > n_thin

    def test_neff_bounded_by_material_indices(self, solver: EffectiveIndexSolver) -> None:
        """n_eff must be between cladding and core indices."""
        n_eff = solver.n_eff()
        assert N_OXIDE < n_eff < N_SILICON

    def test_override_parameters(self) -> None:
        """n_eff should accept width/height overrides."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        n_default = solver.n_eff()
        n_override = solver.n_eff(width=450e-9, height=210e-9)
        # Different geometry should give different n_eff
        assert n_default != n_override


# ---------------------------------------------------------------------------
# TestSensitivityCoefficients
# ---------------------------------------------------------------------------

class TestSensitivityCoefficients:
    """Tests for sensitivity coefficient computation."""

    def test_returns_dataclass(self, solver: EffectiveIndexSolver) -> None:
        """sensitivity() should return SensitivityCoefficients."""
        coeffs = solver.sensitivity()
        assert isinstance(coeffs, SensitivityCoefficients)

    def test_dn_eff_dw_positive(self, solver: EffectiveIndexSolver) -> None:
        """dn_eff/dw should be positive (wider waveguide = higher n_eff)."""
        coeffs = solver.sensitivity()
        assert coeffs.dn_eff_dw > 0

    def test_dn_eff_dh_positive(self, solver: EffectiveIndexSolver) -> None:
        """dn_eff/dh should be positive (thicker silicon = higher n_eff)."""
        coeffs = solver.sensitivity()
        assert coeffs.dn_eff_dh > 0

    def test_dn_eff_dw_typical_magnitude(self, solver: EffectiveIndexSolver) -> None:
        """
        dn_eff/dw should be approximately 1-3 x 10^6 [1/m] = 1-3 x 10^-3 [1/nm].

        Literature: for 500nm x 220nm SOI at 1550nm, dn_eff/dw ~ 1-3e-3 per nm.
        """
        coeffs = solver.sensitivity()
        # Convert to per-nm: dn_eff_dw [1/m] * 1e-9 [m/nm] = dn_eff per nm
        dn_per_nm = coeffs.dn_eff_dw * 1e-9
        assert 0.5e-3 < dn_per_nm < 5e-3, f"Got dn_eff/dw = {dn_per_nm:.4e} per nm"

    def test_finite_difference_convergence(self) -> None:
        """Sensitivity should converge as FD step decreases."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        # Coarse and fine FD steps
        coeffs_coarse = solver.sensitivity(delta_w=5e-9, delta_h=5e-9)
        coeffs_fine = solver.sensitivity(delta_w=0.5e-9, delta_h=0.5e-9)
        coeffs_default = solver.sensitivity(delta_w=1e-9, delta_h=1e-9)

        # All should agree within ~10% (O(h^2) convergence)
        assert_allclose(
            coeffs_default.dn_eff_dw, coeffs_fine.dn_eff_dw, rtol=0.1
        )
        assert_allclose(
            coeffs_default.dn_eff_dh, coeffs_fine.dn_eff_dh, rtol=0.1
        )


# ---------------------------------------------------------------------------
# TestChainRule
# ---------------------------------------------------------------------------

class TestChainRule:
    """Tests for resonance wavelength sensitivity (chain rule)."""

    def test_dlambda_dw_positive(self, solver: EffectiveIndexSolver) -> None:
        """dlambda/dw should be positive (wider -> higher n_eff -> red-shift)."""
        coeffs = solver.sensitivity()
        assert coeffs.dlambda_dw > 0

    def test_dlambda_dh_positive(self, solver: EffectiveIndexSolver) -> None:
        """dlambda/dh should be positive (thicker -> higher n_eff -> red-shift)."""
        coeffs = solver.sensitivity()
        assert coeffs.dlambda_dh > 0

    def test_chain_rule_consistency(self, solver: EffectiveIndexSolver) -> None:
        """dlambda/dw should equal (lambda/n_g) * dn_eff/dw."""
        n_g = 4.2
        coeffs = solver.sensitivity(n_g=n_g)
        wl = solver.waveguide.wavelength

        expected_dlambda_dw = (wl / n_g) * coeffs.dn_eff_dw
        assert_allclose(coeffs.dlambda_dw, expected_dlambda_dw, rtol=1e-10)

        expected_dlambda_dh = (wl / n_g) * coeffs.dn_eff_dh
        assert_allclose(coeffs.dlambda_dh, expected_dlambda_dh, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestNeffMap
# ---------------------------------------------------------------------------

class TestNeffMap:
    """Tests for 2D n_eff map generation."""

    def test_map_shape(self) -> None:
        """n_eff_map should return array of shape (M, N)."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        widths = np.linspace(400e-9, 600e-9, 5)
        heights = np.linspace(200e-9, 250e-9, 4)
        grid = solver.n_eff_map(widths, heights)

        assert grid.shape == (4, 5)

    def test_map_monotonicity_width(self) -> None:
        """n_eff should increase along the width axis."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        widths = np.linspace(400e-9, 600e-9, 5)
        heights = np.array([220e-9])
        grid = solver.n_eff_map(widths, heights)

        # Each row should be monotonically increasing
        row = grid[0, :]
        valid = ~np.isnan(row)
        assert np.all(np.diff(row[valid]) > 0)

    def test_map_monotonicity_height(self) -> None:
        """n_eff should increase along the height axis."""
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)

        widths = np.array([500e-9])
        heights = np.linspace(200e-9, 260e-9, 5)
        grid = solver.n_eff_map(widths, heights)

        col = grid[:, 0]
        valid = ~np.isnan(col)
        assert np.all(np.diff(col[valid]) > 0)


# ---------------------------------------------------------------------------
# TestSingleModeCheck
# ---------------------------------------------------------------------------

class TestSingleModeCheck:
    """Tests for single-mode waveguide verification."""

    def test_standard_500x220_is_single_mode(self) -> None:
        """500nm x 220nm at 1550nm should be single-mode."""
        assert check_single_mode(500e-9, 220e-9) is True

    def test_very_wide_is_multimode(self) -> None:
        """1000nm+ width at 220nm should be multimode."""
        assert check_single_mode(1200e-9, 220e-9) is False


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_narrow_waveguide(self) -> None:
        """Narrow waveguide (300nm) should still give valid n_eff."""
        wg = WaveguideGeometry(width=300e-9, height=220e-9)
        solver = EffectiveIndexSolver(wg)
        n_eff = solver.n_eff()
        assert N_OXIDE < n_eff < N_SILICON

    def test_wide_waveguide(self) -> None:
        """Wide waveguide (800nm) should give higher n_eff than nominal."""
        wg_nominal = WaveguideGeometry(width=500e-9, height=220e-9)
        wg_wide = WaveguideGeometry(width=800e-9, height=220e-9)

        n_nominal = EffectiveIndexSolver(wg_nominal).n_eff()
        n_wide = EffectiveIndexSolver(wg_wide).n_eff()
        assert n_wide > n_nominal
