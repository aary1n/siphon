#!/usr/bin/env python3
"""
SiPhON Phase 0.1 - Analytical Baseline Script

Demonstrates the analytical ring resonator model and thermal tuning calculations.
Run this script to verify the installation and see example outputs.

Usage:
    python scripts/run_analytical.py
"""

import sys
from pathlib import Path

# Add siphon to path (for development without pip install)
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

import numpy as np
from siphon.ring import RingResonator, RingGeometry
from siphon.thermal import ThermalModel, ThermalConfig


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    """Run analytical baseline demonstration."""
    print_header("SiPhON Phase 0.1 - Analytical Baseline")
    print("Silicon Photonics with Open Numerics")
    print("Ring Resonator Analysis & Thermal Tuning Model")

    # Define ring geometry
    print_header("1. Ring Resonator Definition")

    geometry = RingGeometry(
        radius=10e-6,       # 10 μm
        kappa=0.2,          # 20% power coupling
        alpha=2.0,          # 2 dB/cm loss
        n_eff=2.4,          # Effective index
        n_g=4.2,            # Group index
    )

    ring = RingResonator(geometry)
    print(f"\nDevice: {ring}")
    print(f"\nGeometry:")
    print(f"  Circumference: {geometry.circumference * 1e6:.2f} μm")
    print(f"  Self-coupling (r): {geometry.self_coupling:.4f}")
    print(f"  Round-trip loss (a): {geometry.round_trip_loss:.4f}")

    # Calculate metrics at 1550 nm
    wavelength = 1.55e-6  # C-band center
    print_header("2. Performance Metrics @ 1550 nm")

    metrics = ring.metrics(wavelength)
    print(f"\nSpectral Characteristics:")
    print(f"  Free Spectral Range (FSR): {metrics.fsr * 1e9:.3f} nm")
    print(f"  FSR (frequency): {metrics.fsr_ghz:.1f} GHz")
    print(f"  Linewidth (FWHM): {metrics.linewidth * 1e12:.2f} pm")

    print(f"\nQuality Metrics:")
    print(f"  Quality Factor Q: {metrics.quality_factor:.0f}")
    print(f"  Finesse F: {metrics.finesse:.1f}")
    print(f"  Extinction Ratio ER: {metrics.extinction_ratio_db:.1f} dB")

    # Thermal model
    print_header("3. Thermal Tuning Model")

    thermal = ThermalModel(ring)
    thermal_metrics = thermal.metrics(wavelength)

    print(f"\nThermo-optic Parameters:")
    print(f"  dn/dT (silicon): {thermal.config.dn_dt:.2e} K⁻¹")
    print(f"  Thermal resistance: {thermal.config.thermal_resistance:.0f} K/W")

    print(f"\nWavelength Tuning:")
    print(f"  Δλ/ΔT: {thermal_metrics.wavelength_shift_per_kelvin * 1e12:.2f} pm/K")
    print(f"  Tuning efficiency: {thermal_metrics.tuning_efficiency * 1e12:.1f} pm/mW")

    print(f"\nHeater Power Budget:")
    print(f"  Temperature for 1 FSR: {thermal_metrics.temperature_per_fsr:.1f} K")
    print(f"  Power for 1 FSR: {thermal_metrics.power_per_fsr * 1e3:.2f} mW")
    print(f"  Max tuning range: {thermal_metrics.max_tuning_range * 1e9:.2f} nm")

    # Parameter sweep: radius vs FSR
    print_header("4. FSR vs. Radius (Parameter Sweep)")

    radii = [5, 10, 20, 30, 50]  # μm
    print(f"\n{'Radius (μm)':<12} {'FSR (nm)':<12} {'Q':<12} {'P/FSR (mW)':<12}")
    print("-" * 48)

    for R in radii:
        geom = RingGeometry(radius=R * 1e-6, kappa=0.2, alpha=2.0, n_eff=2.4, n_g=4.2)
        ring_temp = RingResonator(geom)
        thermal_temp = ThermalModel(ring_temp)

        fsr = ring_temp.fsr(wavelength) * 1e9
        Q = ring_temp.quality_factor(wavelength)
        power = thermal_temp.power_per_fsr(wavelength) * 1e3

        print(f"{R:<12} {fsr:<12.2f} {Q:<12.0f} {power:<12.2f}")

    # Validation check
    print_header("5. Literature Validation")

    # Expected values
    fsr_expected = (8, 10)  # nm for R=10μm
    dlambda_dT_expected = (0.06, 0.12)  # nm/K

    fsr_computed = metrics.fsr * 1e9
    dlambda_dT_computed = thermal_metrics.wavelength_shift_per_kelvin * 1e9

    fsr_pass = fsr_expected[0] <= fsr_computed <= fsr_expected[1]
    thermal_pass = dlambda_dT_expected[0] <= dlambda_dT_computed <= dlambda_dT_expected[1]

    print(f"\nFSR (R=10μm):")
    print(f"  Computed: {fsr_computed:.2f} nm")
    print(f"  Expected: {fsr_expected[0]}-{fsr_expected[1]} nm")
    print(f"  Status: {'✓ PASS' if fsr_pass else '✗ FAIL'}")

    print(f"\nΔλ/ΔT (silicon @ 1550nm):")
    print(f"  Computed: {dlambda_dT_computed:.3f} nm/K")
    print(f"  Expected: {dlambda_dT_expected[0]}-{dlambda_dT_expected[1]} nm/K")
    print(f"  Status: {'✓ PASS' if thermal_pass else '✗ FAIL'}")

    print_header("Summary")
    if fsr_pass and thermal_pass:
        print("\n✓ All validation checks PASSED")
        print("  Phase 0.1 analytical baseline is ready for use.")
    else:
        print("\n✗ Some validation checks FAILED")
        print("  Review model parameters and assumptions.")

    print("\nNext steps:")
    print("  - Run Jupyter notebook: notebooks/01_analytical_baseline.ipynb")
    print("  - Proceed to Phase 0.2: Variability & Yield Analysis")
    print()


if __name__ == "__main__":
    main()
