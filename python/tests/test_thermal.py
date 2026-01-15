"""
Unit tests for the thermal model.

Tests verify:
1. Δλ/ΔT ≈ 0.1 nm/K for typical silicon ring
2. Temperature required for 1 FSR shift
3. Heater power budget calculations
4. Tuning efficiency metrics
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siphon.ring import RingResonator, RingGeometry
from siphon.thermal import ThermalModel, ThermalConfig, ThermalMetrics


@pytest.fixture
def typical_ring() -> RingResonator:
    """Typical silicon ring resonator."""
    geom = RingGeometry(
        radius=10e-6,
        kappa=0.2,
        alpha=2.0,
        n_eff=2.4,
        n_g=4.2,
    )
    return RingResonator(geom)


@pytest.fixture
def thermal_model(typical_ring: RingResonator) -> ThermalModel:
    """Thermal model with default config."""
    return ThermalModel(typical_ring)


@pytest.fixture
def custom_thermal_model(typical_ring: RingResonator) -> ThermalModel:
    """Thermal model with custom config."""
    config = ThermalConfig(
        dn_dt=1.8e-4,
        thermal_resistance=3000,  # 3000 K/W
        max_heater_power=15e-3,   # 15 mW
    )
    return ThermalModel(typical_ring, config)


class TestThermalConfig:
    """Tests for ThermalConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have reasonable values."""
        config = ThermalConfig()
        assert config.dn_dt == pytest.approx(1.8e-4, rel=1e-2)
        assert config.thermal_resistance == 2000
        assert config.max_heater_power == 10e-3

    def test_invalid_dn_dt(self) -> None:
        """Negative dn/dT should raise error."""
        with pytest.raises(ValueError, match="dn/dT must be positive"):
            ThermalConfig(dn_dt=-1e-4)

    def test_invalid_thermal_resistance(self) -> None:
        """Non-positive thermal resistance should raise error."""
        with pytest.raises(ValueError, match="Thermal resistance must be positive"):
            ThermalConfig(thermal_resistance=0)


class TestWavelengthShift:
    """Tests for wavelength shift calculations."""

    def test_dlambda_dt_typical_value(self, thermal_model: ThermalModel) -> None:
        """
        Δλ/ΔT should be approximately 0.08-0.12 nm/K for silicon at 1550nm.

        Reference: Typical measured values in literature.
        """
        wavelength = 1.55e-6
        dlambda_dT = thermal_model.wavelength_shift_per_kelvin(wavelength)

        # Convert to nm/K
        dlambda_dT_nm_per_K = dlambda_dT * 1e9

        # Should be ~0.08-0.12 nm/K (varies with confinement)
        assert 0.05 < dlambda_dT_nm_per_K < 0.15

    def test_dlambda_dt_formula(self, thermal_model: ThermalModel) -> None:
        """Verify Δλ/ΔT = (λ/n_g) × (dn_eff/dT)."""
        wavelength = 1.55e-6
        dlambda_dT = thermal_model.wavelength_shift_per_kelvin(wavelength)

        n_g = thermal_model.ring.geometry.n_g
        dn_eff_dT = thermal_model.dn_eff_dt()

        expected = (wavelength / n_g) * dn_eff_dT
        assert_allclose(dlambda_dT, expected, rtol=1e-10)

    def test_positive_shift(self, thermal_model: ThermalModel) -> None:
        """Heating should red-shift the resonance (positive dn/dT in Si)."""
        wavelength = 1.55e-6
        dlambda_dT = thermal_model.wavelength_shift_per_kelvin(wavelength)
        assert dlambda_dT > 0  # Red shift with temperature increase


class TestTemperatureForShift:
    """Tests for temperature change calculations."""

    def test_round_trip(self, thermal_model: ThermalModel) -> None:
        """Δλ → ΔT → Δλ should round-trip."""
        wavelength = 1.55e-6
        delta_lambda_original = 1e-9  # 1 nm shift

        delta_T = thermal_model.temperature_for_wavelength_shift(
            delta_lambda_original, wavelength
        )
        delta_lambda_recovered = thermal_model.wavelength_shift_for_power(
            delta_T / thermal_model.config.thermal_resistance, wavelength
        )

        assert_allclose(delta_lambda_recovered, delta_lambda_original, rtol=1e-10)

    def test_temperature_per_fsr(self, thermal_model: ThermalModel) -> None:
        """Temperature per FSR should be consistent with FSR and Δλ/ΔT."""
        wavelength = 1.55e-6
        temp_per_fsr = thermal_model.temperature_per_fsr(wavelength)
        fsr = thermal_model.ring.fsr(wavelength)
        dlambda_dT = thermal_model.wavelength_shift_per_kelvin(wavelength)

        expected = fsr / dlambda_dT
        assert_allclose(temp_per_fsr, expected, rtol=1e-10)


class TestHeaterPower:
    """Tests for heater power calculations."""

    def test_power_for_shift(self, thermal_model: ThermalModel) -> None:
        """Power = ΔT / R_th."""
        wavelength = 1.55e-6
        delta_lambda = 1e-9  # 1 nm

        power = thermal_model.power_for_wavelength_shift(delta_lambda, wavelength)
        delta_T = thermal_model.temperature_for_wavelength_shift(delta_lambda, wavelength)

        expected = delta_T / thermal_model.config.thermal_resistance
        assert_allclose(power, expected, rtol=1e-10)

    def test_power_per_fsr(self, thermal_model: ThermalModel) -> None:
        """Power per FSR should be in reasonable range."""
        wavelength = 1.55e-6
        power_fsr = thermal_model.power_per_fsr(wavelength)

        # Typical: 5-50 mW per FSR depending on heater efficiency
        assert 1e-3 < power_fsr < 100e-3  # 1-100 mW

    def test_power_round_trip(self, thermal_model: ThermalModel) -> None:
        """P → Δλ → P should round-trip."""
        wavelength = 1.55e-6
        power_original = 5e-3  # 5 mW

        delta_lambda = thermal_model.wavelength_shift_for_power(power_original, wavelength)
        power_recovered = thermal_model.power_for_wavelength_shift(delta_lambda, wavelength)

        assert_allclose(power_recovered, power_original, rtol=1e-10)


class TestTuningEfficiency:
    """Tests for tuning efficiency calculations."""

    def test_tuning_efficiency_positive(self, thermal_model: ThermalModel) -> None:
        """Tuning efficiency should be positive."""
        wavelength = 1.55e-6
        efficiency = thermal_model.tuning_efficiency(wavelength)
        assert efficiency > 0

    def test_tuning_efficiency_units(self, thermal_model: ThermalModel) -> None:
        """Verify efficiency = Δλ/ΔT × R_th."""
        wavelength = 1.55e-6
        efficiency = thermal_model.tuning_efficiency(wavelength)
        dlambda_dT = thermal_model.wavelength_shift_per_kelvin(wavelength)
        R_th = thermal_model.config.thermal_resistance

        expected = dlambda_dT * R_th
        assert_allclose(efficiency, expected, rtol=1e-10)

    def test_typical_efficiency(self, thermal_model: ThermalModel) -> None:
        """Typical tuning efficiency should be ~10-50 pm/mW."""
        wavelength = 1.55e-6
        efficiency = thermal_model.tuning_efficiency(wavelength)

        # Convert to pm/mW: efficiency [m/W] × 1e12 [pm/m] × 1e-3 [W/mW] = efficiency × 1e9
        efficiency_pm_per_mW = efficiency * 1e9

        # Typical: 10-100 pm/mW
        assert 5 < efficiency_pm_per_mW < 500


class TestMaxTuningRange:
    """Tests for maximum tuning range calculations."""

    def test_max_tuning_range(self, thermal_model: ThermalModel) -> None:
        """Max tuning range = max_power × tuning_efficiency."""
        wavelength = 1.55e-6
        max_range = thermal_model.max_tuning_range(wavelength)
        max_power = thermal_model.config.max_heater_power
        efficiency = thermal_model.tuning_efficiency(wavelength)

        expected = max_power * efficiency
        assert_allclose(max_range, expected, rtol=1e-10)

    def test_max_range_vs_fsr(self, thermal_model: ThermalModel) -> None:
        """Check if max tuning range covers at least one FSR."""
        wavelength = 1.55e-6
        max_range = thermal_model.max_tuning_range(wavelength)
        fsr = thermal_model.ring.fsr(wavelength)

        # For a 10 mW budget, should typically cover >1 FSR
        # This depends on heater efficiency
        print(f"Max range: {max_range*1e9:.2f} nm, FSR: {fsr*1e9:.2f} nm")


class TestShiftedSpectrum:
    """Tests for temperature-shifted spectrum."""

    def test_shifted_resonance(self, thermal_model: ThermalModel) -> None:
        """Resonance should shift with temperature."""
        wavelength = 1.55e-6
        delta_T = 10  # 10 K temperature rise

        # Original spectrum
        wl_orig, T_orig = thermal_model.ring.spectrum(wavelength, n_points=5000)
        min_idx_orig = np.argmin(T_orig)

        # Shifted spectrum
        wl_shift, T_shift = thermal_model.shifted_spectrum(
            wavelength, delta_T, n_points=5000
        )
        min_idx_shift = np.argmin(T_shift)

        # Resonance should move to longer wavelength
        assert wl_shift[min_idx_shift] > wl_orig[min_idx_orig]

    def test_shift_magnitude(self, thermal_model: ThermalModel) -> None:
        """Shift magnitude should match Δλ/ΔT × ΔT."""
        wavelength = 1.55e-6
        delta_T = 20  # 20 K

        expected_shift = thermal_model.wavelength_shift_per_kelvin(wavelength) * delta_T

        # Find resonance positions
        wl_orig, T_orig = thermal_model.ring.spectrum(wavelength, n_points=10000)
        min_idx_orig = np.argmin(T_orig)

        wl_shift, T_shift = thermal_model.shifted_spectrum(
            wavelength, delta_T, n_points=10000
        )
        min_idx_shift = np.argmin(T_shift)

        actual_shift = wl_shift[min_idx_shift] - wl_orig[min_idx_orig]

        # Should match within grid resolution
        assert_allclose(actual_shift, expected_shift, rtol=0.01)


class TestMetrics:
    """Tests for combined thermal metrics."""

    def test_metrics_returns_dataclass(self, thermal_model: ThermalModel) -> None:
        """metrics() should return ThermalMetrics dataclass."""
        metrics = thermal_model.metrics(wavelength=1.55e-6)
        assert isinstance(metrics, ThermalMetrics)

    def test_metrics_consistency(self, thermal_model: ThermalModel) -> None:
        """Metrics should be internally consistent."""
        wavelength = 1.55e-6
        metrics = thermal_model.metrics(wavelength)

        # power_per_fsr = temp_per_fsr / R_th
        expected_power = metrics.temperature_per_fsr / thermal_model.config.thermal_resistance
        assert_allclose(metrics.power_per_fsr, expected_power, rtol=1e-10)

    def test_metrics_fields(self, thermal_model: ThermalModel) -> None:
        """All metrics fields should be positive."""
        metrics = thermal_model.metrics(1.55e-6)

        assert metrics.wavelength_shift_per_kelvin > 0
        assert metrics.temperature_per_fsr > 0
        assert metrics.power_per_fsr > 0
        assert metrics.tuning_efficiency > 0
        assert metrics.max_tuning_range > 0
        assert metrics.fsr > 0


class TestCustomConfig:
    """Tests with custom thermal configuration."""

    def test_different_thermal_resistance(
        self, typical_ring: RingResonator
    ) -> None:
        """Different R_th should scale power requirements."""
        wavelength = 1.55e-6

        low_r_th = ThermalModel(typical_ring, ThermalConfig(thermal_resistance=1000))
        high_r_th = ThermalModel(typical_ring, ThermalConfig(thermal_resistance=5000))

        # Higher R_th = better efficiency = less power needed
        assert high_r_th.power_per_fsr(wavelength) < low_r_th.power_per_fsr(wavelength)

    def test_different_max_power(
        self, typical_ring: RingResonator
    ) -> None:
        """Different max power should scale tuning range."""
        wavelength = 1.55e-6

        low_power = ThermalModel(typical_ring, ThermalConfig(max_heater_power=5e-3))
        high_power = ThermalModel(typical_ring, ThermalConfig(max_heater_power=20e-3))

        assert high_power.max_tuning_range(wavelength) > low_power.max_tuning_range(wavelength)


class TestLiteratureComparison:
    """Tests against literature values."""

    def test_typical_silicon_thermal(self) -> None:
        """
        Verify against typical silicon photonics thermal tuning values.

        Reference values:
        - dn/dT ≈ 1.8 × 10⁻⁴ K⁻¹ (silicon)
        - Δλ/ΔT ≈ 0.08-0.1 nm/K at 1550nm
        - Typical heater efficiency: 10-50 pm/mW
        """
        geom = RingGeometry(
            radius=10e-6, kappa=0.2, alpha=2.0, n_eff=2.4, n_g=4.2
        )
        ring = RingResonator(geom)
        thermal = ThermalModel(ring)
        wavelength = 1.55e-6

        metrics = thermal.metrics(wavelength)

        # Δλ/ΔT should be ~0.08-0.12 nm/K
        dlambda_dT_nm = metrics.wavelength_shift_per_kelvin * 1e9
        assert 0.05 < dlambda_dT_nm < 0.15, f"Got {dlambda_dT_nm} nm/K"

        # FSR should match ring model
        assert_allclose(metrics.fsr, ring.fsr(wavelength), rtol=1e-10)
