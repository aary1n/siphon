#!/usr/bin/env python3
"""
SiPhON Phase 0.2 - Variability & Yield Analysis Script

Demonstrates the sensitivity model and Monte Carlo yield engine.
Run this script to verify Phase 0.2 installation and see example outputs.

Usage:
    python scripts/run_variability.py
"""

import sys
import time
from pathlib import Path

# Add siphon to path (for development without pip install)
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from siphon.ring import RingResonator, RingGeometry
from siphon.thermal import ThermalModel
from siphon.sensitivity import EffectiveIndexSolver, WaveguideGeometry
from siphon.variability import (
    YieldAnalyzer,
    FabricationConfig,
    MonteCarloConfig,
    quick_yield,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    """Run variability and yield analysis demonstration."""
    print_header("SiPhON Phase 0.2 - Variability & Yield Analysis")
    print("Silicon Photonics with Open Numerics")
    print("Process Variation -> Sensitivity -> Monte Carlo -> Yield")

    # -------------------------------------------------------------------
    # 1. Effective Index Method
    # -------------------------------------------------------------------
    print_header("1. Effective Index Method (EIM)")

    wg = WaveguideGeometry(width=500e-9, height=220e-9)
    solver = EffectiveIndexSolver(wg)
    n_eff = solver.n_eff()

    print(f"\nWaveguide: {solver}")
    print(f"  Width: 500 nm")
    print(f"  Height: 220 nm")
    print(f"  Wavelength: 1550 nm")
    print(f"  n_core (Si): {wg.n_core}")
    print(f"  n_clad (SiO2): {wg.n_clad}")
    print(f"\nEffective Index (EIM): n_eff = {n_eff:.6f}")

    # -------------------------------------------------------------------
    # 2. Sensitivity Coefficients
    # -------------------------------------------------------------------
    print_header("2. Sensitivity Coefficients")

    n_g = 4.2  # Group index for silicon at 1550nm
    sens = solver.sensitivity(n_g=n_g)

    # Convert to per-nm for readability
    dn_dw_per_nm = sens.dn_eff_dw * 1e-9
    dn_dh_per_nm = sens.dn_eff_dh * 1e-9
    dl_dw_nm = sens.dlambda_dw * 1e9  # nm per nm
    dl_dh_nm = sens.dlambda_dh * 1e9  # nm per nm

    print(f"\nMethod: {sens.method}")
    print(f"Group index (n_g): {n_g}")
    print(f"\nGeometry Sensitivities:")
    print(f"  dn_eff/dw = {dn_dw_per_nm:.4e} per nm width change")
    print(f"  dn_eff/dh = {dn_dh_per_nm:.4e} per nm height change")
    print(f"\nResonance Wavelength Sensitivities (chain rule):")
    print(f"  d(lambda)/dw = {dl_dw_nm:.4f} nm per nm width change")
    print(f"  d(lambda)/dh = {dl_dh_nm:.4f} nm per nm height change")

    # -------------------------------------------------------------------
    # 3. Fabrication Tolerances
    # -------------------------------------------------------------------
    print_header("3. Fabrication Tolerance Definition")

    fab = FabricationConfig()
    print(f"\nProcess Variation Model (Gaussian):")
    print(f"  Width:  w ~ N({fab.w_nominal*1e9:.0f} nm, sigma={fab.sigma_w*1e9:.0f} nm)")
    print(f"  Height: h ~ N({fab.h_nominal*1e9:.0f} nm, sigma={fab.sigma_h*1e9:.0f} nm)")
    print(f"  Correlation: {fab.correlation}")

    # -------------------------------------------------------------------
    # 4. Monte Carlo Yield Analysis
    # -------------------------------------------------------------------
    print_header("4. Monte Carlo Yield Analysis")

    # Build ring and thermal models using EIM n_eff
    geom = RingGeometry(
        radius=10e-6, kappa=0.2, alpha=2.0,
        n_eff=sens.n_eff, n_g=n_g,
    )
    ring = RingResonator(geom)
    thermal = ThermalModel(ring)
    mc = MonteCarloConfig(n_samples=10_000, seed=42, max_heater_power=10e-3)
    analyzer = YieldAnalyzer(ring, thermal, sens, fab, mc)

    print(f"\nConfiguration: {analyzer}")
    print(f"Heater power budget: {mc.max_heater_power*1e3:.0f} mW")

    start = time.perf_counter()
    result = analyzer.run()
    elapsed = time.perf_counter() - start

    print(f"\nResults ({result.n_samples} samples, {elapsed*1e3:.1f} ms):")
    print(f"  Yield: {result.yield_percent:.1f}%")
    print(f"  Mean heater power: {result.mean_heater_power*1e3:.2f} mW")
    print(f"  Std heater power: {result.std_heater_power*1e3:.2f} mW")
    print(f"  95th percentile: {result.p95_heater_power*1e3:.2f} mW")

    # -------------------------------------------------------------------
    # 5. Cost of Variance Analysis
    # -------------------------------------------------------------------
    print_header("5. Cost of Variance Analysis")

    print(f"\nComparing fabrication tolerance improvements:")
    print(f"  Power budget: {mc.max_heater_power*1e3:.0f} mW\n")

    sigma_scenarios = [
        ("Tight (sigma_w=5nm)", 5e-9, 2.5e-9),
        ("Standard (sigma_w=10nm)", 10e-9, 5e-9),
        ("Loose (sigma_w=15nm)", 15e-9, 7.5e-9),
    ]

    print(f"{'Scenario':<28} {'Yield':<10} {'Mean P (mW)':<14} {'P95 (mW)':<12}")
    print("-" * 64)

    for label, sw, sh in sigma_scenarios:
        r = quick_yield(
            n_eff=sens.n_eff, n_g=n_g,
            sigma_w=sw, sigma_h=sh,
            max_power=mc.max_heater_power,
            n_samples=10_000, seed=42,
        )
        print(f"{label:<28} {r.yield_percent:<10.1f} {r.mean_heater_power*1e3:<14.2f} {r.p95_heater_power*1e3:<12.2f}")

    # -------------------------------------------------------------------
    # 6. Validation
    # -------------------------------------------------------------------
    print_header("6. Validation Checks")

    # Check n_eff in expected range
    neff_pass = 2.2 < n_eff < 2.6
    print(f"\nn_eff (500nm x 220nm Si @ 1550nm):")
    print(f"  Computed: {n_eff:.6f}")
    print(f"  Expected: 2.2 - 2.6 (EIM approximation)")
    print(f"  Status: {'[PASS]' if neff_pass else '[FAIL]'}")

    # Check sensitivities positive
    sens_pass = sens.dn_eff_dw > 0 and sens.dn_eff_dh > 0
    print(f"\nSensitivity signs:")
    print(f"  dn_eff/dw > 0: {sens.dn_eff_dw > 0}")
    print(f"  dn_eff/dh > 0: {sens.dn_eff_dh > 0}")
    print(f"  Status: {'[PASS]' if sens_pass else '[FAIL]'}")

    # Check MC performance
    perf_pass = elapsed < 10.0
    print(f"\nMC performance (10,000 samples):")
    print(f"  Time: {elapsed*1e3:.1f} ms")
    print(f"  Budget: <10,000 ms")
    print(f"  Status: {'[PASS]' if perf_pass else '[FAIL]'}")

    # Check yield is reasonable
    yield_pass = 0.1 < result.yield_fraction <= 1.0
    print(f"\nYield reasonableness:")
    print(f"  Yield: {result.yield_percent:.1f}%")
    print(f"  Expected: 10-100% for typical parameters")
    print(f"  Status: {'[PASS]' if yield_pass else '[FAIL]'}")

    print_header("Summary")
    all_pass = neff_pass and sens_pass and perf_pass and yield_pass
    if all_pass:
        print("\n[PASS] All validation checks PASSED")
        print("  Phase 0.2 variability engine is ready for use.")
    else:
        print("\n[FAIL] Some validation checks FAILED")
        print("  Review model parameters and assumptions.")

    print("\nNext steps:")
    print("  - Run Jupyter notebook: notebooks/02_sensitivity_maps.ipynb")
    print("  - Run Jupyter notebook: notebooks/03_yield_analysis.ipynb")
    print("  - Proceed to Phase 0.3: C++ Computational Core")
    print()


if __name__ == "__main__":
    main()
