# SiPhON - Silicon Photonics with Open Numerics

**Version:** 0.1.0-dev (Phase 0.1: Core Physics Baseline)

A rigorous, numerically-driven simulation framework for silicon photonic ring resonators, demonstrating the "Physics → Numerics → Yield" workflow for high quality engineering analysis.

---

## Project Status

✅ **Phase 0.1 Complete**: Analytical baseline with all-pass ring transfer function and thermal tuning model

🚧 **Phase 0.2 Next**: Variability engine and Monte Carlo yield analysis

---

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install SiPhON in development mode
pip install -e ./python

# Install development dependencies (optional)
pip install -e "./python[dev,notebook]"
```

### Run Tests

```bash
cd python
pytest tests/ -v
```

**Expected Result:** ✅ All 48 tests pass

### Run Analytical Baseline Demo

```bash
python scripts/run_analytical.py
```

**Output includes:**
- Ring resonator metrics (FSR, Q, ER, Finesse)
- Thermal tuning parameters (wavelength shift, heater power budget)
- Literature validation checks

### Explore in Jupyter

```bash
jupyter notebook notebooks/01_analytical_baseline.ipynb
```

**Includes:**
- Interactive transmission spectra
- Parameter sweep studies (FSR vs. radius, Q vs. loss, ER vs. coupling)
- Thermal tuning visualizations
- Literature comparisons

---

## What's Implemented (Phase 0.1)

### 1. Analytical Ring Model (`siphon.ring`)

**All-pass transfer function:**

```
T(φ) = (a² - 2ra·cos(φ) + r²) / (1 - 2ra·cos(φ) + (ra)²)
```

**Key metrics calculated:**
- Free Spectral Range (FSR): λ² / (n_g · L)
- Quality Factor (Q): λ / Δλ
- Extinction Ratio (ER): T_max / T_min
- Finesse (F): FSR / Δλ

**Example:**

```python
from siphon.ring import RingResonator, RingGeometry

geom = RingGeometry(
    radius=10e-6,   # 10 μm
    kappa=0.2,      # 20% coupling
    alpha=2.0,      # 2 dB/cm loss
    n_eff=2.4,
    n_g=4.2
)

ring = RingResonator(geom)
metrics = ring.metrics(wavelength=1.55e-6)

print(f"FSR = {metrics.fsr * 1e9:.2f} nm")
print(f"Q = {metrics.quality_factor:.0f}")
```

### 2. Thermal Tuning Model (`siphon.thermal`)

**Thermo-optic wavelength shift:**
- dn/dT ≈ 1.8 × 10⁻⁴ K⁻¹ for silicon
- Δλ/ΔT = (λ / n_g) × (dn_eff/dT) ≈ 0.06-0.10 nm/K

**Heater power budget:**
- P = ΔT / R_th
- Calculates power required to tune by 1 FSR

**Example:**

```python
from siphon.thermal import ThermalModel

thermal = ThermalModel(ring)
metrics = thermal.metrics(wavelength=1.55e-6)

print(f"Wavelength shift: {metrics.wavelength_shift_per_kelvin * 1e12:.1f} pm/K")
print(f"Power per FSR: {metrics.power_per_fsr * 1e3:.1f} mW")
```

### 3. Comprehensive Test Suite

**48 unit tests** covering:
- Ring geometry validation
- FSR analytical formula verification
- Q vs. loss/coupling dependencies
- Transmission spectrum bounds
- Thermal model accuracy
- Literature comparison

**Run tests:**
```bash
pytest python/tests/ -v
```

---

## Architecture Principles

1. **Performance-critical numerics in C++** (Phase 0.3+): Custom 2D FDE solver
2. **Zero-copy Python bindings** (Phase 0.3+): `pybind11` for C++↔Python
3. **Reproducibility first**: Version-locked dependencies, documented assumptions
4. **Rigor over speed**: Validate convergence, quantify uncertainty

---

## File Structure

```
SIPHON/
├── python/
│   ├── siphon/
│   │   ├── __init__.py
│   │   ├── ring.py          # Analytical ring model
│   │   └── thermal.py       # Thermal tuning model
│   ├── tests/
│   │   ├── test_ring.py     # Ring model tests (24 tests)
│   │   └── test_thermal.py  # Thermal model tests (24 tests)
│   └── pyproject.toml       # Package configuration
├── notebooks/
│   └── 01_analytical_baseline.ipynb  # Interactive analysis
├── scripts/
│   └── run_analytical.py    # Demo script
├── ROADMAP.md               # Development roadmap
├── TODO.md                  # Task tracking
└── CLAUDE.md                # Project manifest & MCP protocols
```

---

## Validation Results

### Literature Comparison

| Metric | Computed (R=10μm) | Literature Range | Status |
|--------|-------------------|------------------|--------|
| FSR | 9.10 nm | 8-10 nm | ✅ PASS |
| Q | 25,308 | 10,000-100,000 | ✅ PASS |
| Δλ/ΔT | 0.060 nm/K | 0.06-0.12 nm/K | ✅ PASS |

### Test Coverage

- **Ring Model:** 24/24 tests pass
- **Thermal Model:** 24/24 tests pass
- **Total:** 48/48 tests pass ✅

---

## Next Steps: Phase 0.2 - Variability & Yield

**Objectives:**
1. Define fabrication tolerance distributions (σ_w, σ_h)
2. Implement Monte Carlo sampling engine
3. Map process variation → n_eff variation → heater power distribution
4. Calculate yield metric: % devices tunable within power budget

**Deliverables:**
- `siphon.yield` module with Monte Carlo engine
- Sensitivity maps: ∂n_eff/∂w, ∂n_eff/∂h
- Yield vs. tolerance curves
- Notebook: `02_yield_analysis.ipynb`

---

## Dependencies

**Core:**
- Python ≥ 3.11
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

**Development:**
- pytest ≥ 7.0
- jupyter ≥ 1.0

**Future (Phase 0.3):**
- CMake ≥ 3.20
- Eigen ≥ 3.4
- Spectra (eigenvalue solver)
- pybind11

---

## References

1. Bogaerts, W. et al., "Silicon microring resonators," *Laser Photon. Rev.* **6**, 47-73 (2012)
2. Yariv, A., "Universal relations for coupling of optical power," *Electron. Lett.* **36**, 321-322 (2000)
3. Cocorullo, G. et al., "Thermo-optic coefficient of silicon," *J. Appl. Phys.* **74**, 3271 (1993)

---

## License

(TBD - academic/open-source pending university IP review)

---

## Contributing

Key principles for SiPhON development:

1. ✅ **Verify numerics first**: Grid convergence, solver residuals, analytical limits
2. 📝 **Document assumptions**: What physics is included? What's neglected?
3. 🔁 **Reproducibility**: Lock versions, seed RNGs, save full parameter sets
4. 🎯 **Correctness > speed**: Rigor before optimization

---

**Last Updated:** 2026-01-16
**Architecture Version:** SiPhON v0.1-dev (Core Physics Baseline)
