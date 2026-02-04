# SiPhON

Silicon Photonics with Open Numerics - a simulation framework for silicon photonic ring resonators.

## Overview

SiPhON implements the analytical transfer function for all-pass ring resonators with thermal tuning, following the Physics → Numerics → Yield workflow. The current release (v0.1-dev) provides the core physics baseline; Monte Carlo yield analysis and FDE mode solving are planned for subsequent phases.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e ./python
```

For development (tests and notebooks):
```bash
pip install -e "./python[dev,notebook]"
```

## Usage

### Ring resonator analysis

```python
from siphon.ring import RingResonator, RingGeometry

geom = RingGeometry(
    radius=10e-6,   # 10 μm
    kappa=0.2,      # power coupling coefficient
    alpha=2.0,      # propagation loss (dB/cm)
    n_eff=2.4,
    n_g=4.2
)

ring = RingResonator(geom)
metrics = ring.metrics(wavelength=1.55e-6)

print(f"FSR = {metrics.fsr * 1e9:.2f} nm")
print(f"Q = {metrics.quality_factor:.0f}")
```

### Thermal tuning

```python
from siphon.thermal import ThermalModel

thermal = ThermalModel(ring)
metrics = thermal.metrics(wavelength=1.55e-6)

print(f"Wavelength shift: {metrics.wavelength_shift_per_kelvin * 1e12:.1f} pm/K")
print(f"Power per FSR: {metrics.power_per_fsr * 1e3:.1f} mW")
```

### Running the demo

```bash
python scripts/run_analytical.py
```

### Notebooks

The `notebooks/` directory contains interactive analyses. Start with `01_analytical_baseline.ipynb` for transmission spectra, parameter sweeps, and literature comparisons.

## Physics

The all-pass ring transfer function is

```
T(φ) = (a² - 2ra·cos(φ) + r²) / (1 - 2ra·cos(φ) + (ra)²)
```

where `a` is the round-trip amplitude transmission, `r` is the self-coupling coefficient, and `φ` is the round-trip phase. Standard metrics (FSR, Q, extinction ratio, finesse) follow from this expression.

Thermal tuning uses the thermo-optic coefficient of silicon (dn/dT ≈ 1.8 × 10⁻⁴ K⁻¹) to compute wavelength shift per kelvin and heater power budgets.

## Validation

Computed values for a 10 μm radius ring at 1550 nm:

| Metric | Computed | Literature |
|--------|----------|------------|
| FSR | 9.10 nm | 8–10 nm |
| Q | 25,308 | 10⁴–10⁵ |
| Δλ/ΔT | 0.060 nm/K | 0.06–0.12 nm/K |

The test suite (48 tests) verifies analytical formulae, physical bounds, and literature agreement:

```bash
cd python && pytest tests/ -v
```

## Project structure

```
python/siphon/       Core modules (ring.py, thermal.py)
python/tests/        Unit tests
notebooks/           Jupyter notebooks for interactive analysis
scripts/             Command-line demos
```

## Roadmap

Phase 0.1 (current): Analytical baseline with thermal tuning  
Phase 0.2: Monte Carlo variability engine and yield analysis  
Phase 0.3: 2D finite-difference eigenmode solver in C++ with pybind11 bindings

## References

W. Bogaerts et al., "Silicon microring resonators," Laser Photon. Rev. 6, 47–73 (2012)  
A. Yariv, "Universal relations for coupling of optical power," Electron. Lett. 36, 321–322 (2000)  
G. Cocorullo et al., "Thermo-optic coefficient of silicon," J. Appl. Phys. 74, 3271 (1993)

## License

MIT
