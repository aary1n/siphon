"""
Unit tests for the Monte Carlo variability engine and yield analysis.

Tests verify:
1. FabricationConfig validation and covariance matrix construction
2. MonteCarloConfig validation and seed reproducibility
3. YieldAnalyzer produces valid results with correct shapes
4. Yield decreases with increasing sigma and increases with power budget
5. Performance: 10k samples in <10 seconds
6. Physical reasonableness of yield values
"""

import time

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siphon.ring import RingResonator, RingGeometry
from siphon.thermal import ThermalModel
from siphon.sensitivity import EffectiveIndexSolver, WaveguideGeometry, SensitivityCoefficients
from siphon.variability import (
    FabricationConfig,
    MonteCarloConfig,
    YieldResult,
    YieldAnalyzer,
    quick_yield,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sensitivity_coefficients() -> SensitivityCoefficients:
    """Sensitivity coefficients for typical 500nm x 220nm SOI."""
    wg = WaveguideGeometry(width=500e-9, height=220e-9)
    solver = EffectiveIndexSolver(wg)
    return solver.sensitivity(n_g=4.2)


@pytest.fixture
def ring(sensitivity_coefficients: SensitivityCoefficients) -> RingResonator:
    """Ring resonator using EIM-derived n_eff."""
    geom = RingGeometry(
        radius=10e-6,
        kappa=0.2,
        alpha=2.0,
        n_eff=sensitivity_coefficients.n_eff,
        n_g=4.2,
    )
    return RingResonator(geom)


@pytest.fixture
def thermal(ring: RingResonator) -> ThermalModel:
    """Thermal model for the ring."""
    return ThermalModel(ring)


@pytest.fixture
def analyzer(
    ring: RingResonator,
    thermal: ThermalModel,
    sensitivity_coefficients: SensitivityCoefficients,
) -> YieldAnalyzer:
    """Yield analyzer with default configuration."""
    return YieldAnalyzer(ring, thermal, sensitivity_coefficients)


# ---------------------------------------------------------------------------
# TestFabricationConfig
# ---------------------------------------------------------------------------

class TestFabricationConfig:
    """Tests for FabricationConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have standard SOI tolerances."""
        fab = FabricationConfig()
        assert fab.w_nominal == 500e-9
        assert fab.h_nominal == 220e-9
        assert fab.sigma_w == 10e-9
        assert fab.sigma_h == 5e-9
        assert fab.correlation == 0.0

    def test_invalid_nominal_width(self) -> None:
        """Non-positive nominal width should raise."""
        with pytest.raises(ValueError, match="Nominal width"):
            FabricationConfig(w_nominal=0)

    def test_invalid_sigma(self) -> None:
        """Negative sigma should raise."""
        with pytest.raises(ValueError, match="Width sigma"):
            FabricationConfig(sigma_w=-1e-9)
        with pytest.raises(ValueError, match="Height sigma"):
            FabricationConfig(sigma_h=-1e-9)

    def test_invalid_correlation(self) -> None:
        """Correlation outside [-1, 1] should raise."""
        with pytest.raises(ValueError, match="Correlation"):
            FabricationConfig(correlation=1.5)
        with pytest.raises(ValueError, match="Correlation"):
            FabricationConfig(correlation=-1.1)

    def test_covariance_matrix_shape(self) -> None:
        """Covariance matrix should be 2x2 symmetric."""
        fab = FabricationConfig()
        cov = fab.covariance_matrix
        assert cov.shape == (2, 2)
        assert_allclose(cov, cov.T)  # Symmetric

    def test_covariance_matrix_values(self) -> None:
        """Covariance matrix diagonal should be sigma^2."""
        fab = FabricationConfig(sigma_w=10e-9, sigma_h=5e-9, correlation=0.3)
        cov = fab.covariance_matrix
        assert_allclose(cov[0, 0], (10e-9)**2)
        assert_allclose(cov[1, 1], (5e-9)**2)
        assert_allclose(cov[0, 1], 0.3 * 10e-9 * 5e-9)


# ---------------------------------------------------------------------------
# TestMonteCarloConfig
# ---------------------------------------------------------------------------

class TestMonteCarloConfig:
    """Tests for MonteCarloConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should match TODO.md spec."""
        mc = MonteCarloConfig()
        assert mc.n_samples == 10_000
        assert mc.seed == 42
        assert mc.max_heater_power == 10e-3

    def test_invalid_n_samples(self) -> None:
        """Non-positive n_samples should raise."""
        with pytest.raises(ValueError, match="n_samples"):
            MonteCarloConfig(n_samples=0)

    def test_invalid_max_power(self) -> None:
        """Non-positive max_heater_power should raise."""
        with pytest.raises(ValueError, match="max_heater_power"):
            MonteCarloConfig(max_heater_power=0)


# ---------------------------------------------------------------------------
# TestYieldAnalyzer
# ---------------------------------------------------------------------------

class TestYieldAnalyzer:
    """Tests for the core yield analysis engine."""

    def test_run_returns_yield_result(self, analyzer: YieldAnalyzer) -> None:
        """run() should return a YieldResult dataclass."""
        result = analyzer.run()
        assert isinstance(result, YieldResult)

    def test_yield_between_0_and_1(self, analyzer: YieldAnalyzer) -> None:
        """Yield fraction should be in [0, 1]."""
        result = analyzer.run()
        assert 0 <= result.yield_fraction <= 1
        assert 0 <= result.yield_percent <= 100
        assert_allclose(result.yield_percent, result.yield_fraction * 100)

    def test_yield_decreases_with_increasing_sigma(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity_coefficients: SensitivityCoefficients,
    ) -> None:
        """Yield should decrease when fabrication tolerance increases."""
        results = []
        for sigma_w in [3e-9, 10e-9, 20e-9]:
            fab = FabricationConfig(sigma_w=sigma_w, sigma_h=sigma_w / 2)
            analyzer = YieldAnalyzer(ring, thermal, sensitivity_coefficients, fab)
            results.append(analyzer.run())

        # Yield should decrease with increasing sigma
        assert results[0].yield_fraction >= results[1].yield_fraction
        assert results[1].yield_fraction >= results[2].yield_fraction

    def test_yield_increases_with_power_budget(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity_coefficients: SensitivityCoefficients,
    ) -> None:
        """Yield should increase when power budget is relaxed."""
        results = []
        for max_power in [5e-3, 10e-3, 50e-3]:
            mc = MonteCarloConfig(max_heater_power=max_power)
            analyzer = YieldAnalyzer(ring, thermal, sensitivity_coefficients, mc_config=mc)
            results.append(analyzer.run())

        assert results[0].yield_fraction <= results[1].yield_fraction
        assert results[1].yield_fraction <= results[2].yield_fraction

    def test_zero_sigma_gives_high_yield(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity_coefficients: SensitivityCoefficients,
    ) -> None:
        """Zero process variation should give ~100% yield."""
        fab = FabricationConfig(sigma_w=1e-15, sigma_h=1e-15)
        analyzer = YieldAnalyzer(ring, thermal, sensitivity_coefficients, fab)
        result = analyzer.run()
        assert result.yield_fraction > 0.99

    def test_heater_powers_non_negative(self, analyzer: YieldAnalyzer) -> None:
        """All heater powers should be non-negative."""
        result = analyzer.run()
        assert np.all(result.heater_powers >= 0)

    def test_sample_count_matches_config(self, analyzer: YieldAnalyzer) -> None:
        """Output array lengths should match n_samples."""
        result = analyzer.run()
        n = analyzer.mc_config.n_samples
        assert result.n_samples == n
        assert len(result.heater_powers) == n
        assert len(result.wavelength_shifts) == n
        assert len(result.delta_n_eff) == n
        assert len(result.width_samples) == n
        assert len(result.height_samples) == n

    def test_reproducibility_with_seed(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity_coefficients: SensitivityCoefficients,
    ) -> None:
        """Same seed should produce identical results."""
        mc = MonteCarloConfig(seed=123)
        a1 = YieldAnalyzer(ring, thermal, sensitivity_coefficients, mc_config=mc)
        a2 = YieldAnalyzer(ring, thermal, sensitivity_coefficients, mc_config=mc)

        r1 = a1.run()
        r2 = a2.run()

        assert_allclose(r1.heater_powers, r2.heater_powers)
        assert r1.yield_fraction == r2.yield_fraction


# ---------------------------------------------------------------------------
# TestPerformance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Tests for performance requirements."""

    def test_10k_samples_under_10_seconds(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity_coefficients: SensitivityCoefficients,
    ) -> None:
        """10,000 samples should complete in <10 seconds (acceptance criterion)."""
        mc = MonteCarloConfig(n_samples=10_000)
        analyzer = YieldAnalyzer(ring, thermal, sensitivity_coefficients, mc_config=mc)

        start = time.perf_counter()
        result = analyzer.run()
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"MC took {elapsed:.2f}s, exceeds 10s budget"
        assert result.n_samples == 10_000

    def test_output_shapes_correct(self, analyzer: YieldAnalyzer) -> None:
        """All output arrays should be 1D with correct length."""
        result = analyzer.run()
        n = analyzer.mc_config.n_samples

        assert result.heater_powers.ndim == 1
        assert result.wavelength_shifts.ndim == 1
        assert result.delta_n_eff.ndim == 1
        assert result.width_samples.shape == (n,)
        assert result.height_samples.shape == (n,)


# ---------------------------------------------------------------------------
# TestSweepTolerance
# ---------------------------------------------------------------------------

class TestSweepTolerance:
    """Tests for tolerance sweep functionality."""

    def test_sweep_returns_arrays(self, analyzer: YieldAnalyzer) -> None:
        """sweep_tolerance should return three arrays."""
        sigmas, yields, powers = analyzer.sweep_tolerance(
            sigma_w_range=np.linspace(2e-9, 15e-9, 5)
        )
        assert len(sigmas) == 5
        assert len(yields) == 5
        assert len(powers) == 5

    def test_yield_monotonically_decreases(self, analyzer: YieldAnalyzer) -> None:
        """Yield should generally decrease as sigma increases."""
        sigmas, yields, powers = analyzer.sweep_tolerance(
            sigma_w_range=np.array([2e-9, 5e-9, 10e-9, 15e-9, 20e-9])
        )
        # Allow small non-monotonicity due to MC noise, but overall trend should be down
        assert yields[0] >= yields[-1]

    def test_mean_power_increases_with_sigma(self, analyzer: YieldAnalyzer) -> None:
        """Mean heater power should increase with larger tolerances."""
        sigmas, yields, powers = analyzer.sweep_tolerance(
            sigma_w_range=np.array([2e-9, 10e-9, 20e-9])
        )
        assert powers[-1] > powers[0]


# ---------------------------------------------------------------------------
# TestQuickYield
# ---------------------------------------------------------------------------

class TestQuickYield:
    """Tests for the quick_yield convenience function."""

    def test_returns_yield_result(self) -> None:
        """quick_yield should return a YieldResult."""
        result = quick_yield(n_samples=1000, seed=42)
        assert isinstance(result, YieldResult)

    def test_matches_explicit_construction(self) -> None:
        """quick_yield should produce same results as explicit setup."""
        # Quick path
        r_quick = quick_yield(
            radius=10e-6, kappa=0.2, alpha=2.0, n_g=4.2,
            sigma_w=10e-9, sigma_h=5e-9,
            n_samples=5000, seed=42,
        )

        # Explicit path
        wg = WaveguideGeometry(width=500e-9, height=220e-9)
        sens = EffectiveIndexSolver(wg).sensitivity(n_g=4.2)
        geom = RingGeometry(radius=10e-6, kappa=0.2, alpha=2.0, n_eff=sens.n_eff, n_g=4.2)
        ring = RingResonator(geom)
        thermal = ThermalModel(ring)
        fab = FabricationConfig(sigma_w=10e-9, sigma_h=5e-9)
        mc = MonteCarloConfig(n_samples=5000, seed=42)
        r_explicit = YieldAnalyzer(ring, thermal, sens, fab, mc).run()

        assert_allclose(r_quick.heater_powers, r_explicit.heater_powers)
        assert r_quick.yield_fraction == r_explicit.yield_fraction


# ---------------------------------------------------------------------------
# TestPhysicalReasonableness
# ---------------------------------------------------------------------------

class TestPhysicalReasonableness:
    """Tests for physically reasonable output values."""

    def test_typical_yield_range(self) -> None:
        """
        For typical SOI tolerances (sigma_w=10nm, sigma_h=5nm)
        with 10mW budget, yield should be in a reasonable range.
        """
        result = quick_yield(
            sigma_w=10e-9, sigma_h=5e-9, max_power=10e-3,
            n_samples=10_000, seed=42,
        )
        # Yield should be between 10% and 100% for these parameters
        assert 0.1 < result.yield_fraction <= 1.0, (
            f"Yield {result.yield_percent:.1f}% outside reasonable range"
        )

    def test_mean_power_reasonable(self) -> None:
        """Mean heater power should be in mW range for typical tolerances."""
        result = quick_yield(n_samples=10_000, seed=42)
        # Mean power should be in 0.1-50 mW range
        assert 0.1e-3 < result.mean_heater_power < 50e-3, (
            f"Mean power {result.mean_heater_power*1e3:.2f}mW outside range"
        )

    def test_statistics_consistency(self) -> None:
        """Statistical fields should be consistent with arrays."""
        result = quick_yield(n_samples=10_000, seed=42)

        assert_allclose(result.mean_heater_power, np.mean(result.heater_powers), rtol=1e-10)
        assert_allclose(result.std_heater_power, np.std(result.heater_powers), rtol=1e-10)
        assert_allclose(
            result.p95_heater_power,
            np.percentile(result.heater_powers, 95),
            rtol=1e-10,
        )
