# SiPhON - Silicon Photonics with Open Numerics

## Project Manifest

**Version:** 0.1-dev (Core Physics Baseline)
**Status:** Architecture Phase
**Target:** Rigorous Portfolio Artifact - Numerical Yield Analysis for Silicon Photonics

---

## Architecture: "Hybrid Simulation Pipeline"

SiPhON is a **high-performance, numerically rigorous simulation framework** for silicon photonic ring resonators, demonstrating the "Physics → Numerics → Yield" workflow.

### Core Principles

1. **Performance-Critical Numerics in C++**: Custom 2D FDE solver eliminates black-box dependencies.
2. **Zero-Copy Python Bindings**: `pybind11`/`nanobind` for seamless C++↔Python interop.
3. **Reproducibility First**: CMake builds, version-locked dependencies, Jupyter reproducibility.
4. **Rigor Over Speed**: Document assumptions, validate convergence, quantify uncertainty.

### Analysis Pipeline

**Goal:** Connect process variation → effective index variation → thermal power budget

Flow:
- Analytical ring model (FSR, Q, ER) → baseline physics
- C++ mode solver (2D Helmholtz) → n_eff(w, h) sensitivity maps
- Monte Carlo sampling (Python) → fabrication tolerance → heater power distribution
- Yield metric: % devices tunable within max heater power budget

---

## Technology Stack

### C++ (C++17)
- **Build System:** CMake (modern, cross-platform)
- **Linear Algebra:** Eigen (sparse matrices, zero-copy views)
- **Eigenvalue Solver:** Spectra (Arnoldi iteration for sparse systems)
- **Python Binding:** `pybind11` (zero-copy NumPy↔Eigen interface)

### Python (3.11+)
- **Purpose:** Analysis orchestration, visualization, Monte Carlo
- **Dependencies:**
  - `numpy` (array operations)
  - `scipy` (analytical models, optimization)
  - `matplotlib` (plotting)
  - `jupyter` (reproducible reporting)
- **Interface:** Native C++ extension via `pybind11`

---

# MCP TOOLING PROTOCOLS

## Overview
SiPhON development requires precision in numerical methods and library usage. **Never guess** API behavior, numerical solver configurations, or project structure. Use MCP tools to fetch authoritative information.

---

## 1. Context7 (Documentation Fetching)

### When to Use
**TRIGGER on these patterns:**
- Mentioning specific library versions (e.g., "Eigen 3.4", "Spectra", "pybind11")
- Working with C++ template libraries (`Eigen::SparseMatrix`, `Spectra::GenEigsSolver`)
- Python scientific stack (`numpy`, `scipy.sparse`, `matplotlib`)
- Questions about API signatures, numerical stability, or version-specific behavior

**ACTION:**
```
Explicitly call: use context7 to fetch docs for [Library + Version]
```

**Examples:**
- "How do I configure Spectra for shift-invert mode?" → `context7: Spectra eigenvalue solver documentation`
- "What's the pybind11 syntax for zero-copy arrays?" → `context7: pybind11 NumPy integration`
- "Eigen sparse matrix assembly patterns?" → `context7: Eigen sparse matrix documentation`

**CONSTRAINT:**
- Do NOT infer template signatures or solver parameters from memory
- If uncertain about numerical stability or convergence → **fetch docs first**

**FALLBACK:**
- If Context7 fails or returns insufficient info → Ask user for official docs URL
- Then use `read` tool to fetch the specific page

---

## 2. Filesystem (Code Navigation & Analysis)

### When to Use
**TRIGGER on these patterns:**
- "Where is [solver/module/class] implemented?"
- "Show me the current [CMakeLists/Python binding/solver code]"
- Before suggesting refactors or architectural changes
- When debugging requires seeing actual file structure

**ACTION:**
```
1. Use filesystem to list directories and locate relevant files
2. Read specific files to understand current implementation
3. Only then propose changes based on ACTUAL code, not assumptions
```

**Examples:**
- "How is the FDE solver structured?" → List src/ → Read solver files
- "Where are Python bindings defined?" → Locate pybind11 wrapper files
- "Current CMake configuration?" → Read CMakeLists.txt files

**CONSTRAINT:**
- Do NOT suggest code changes without first reading existing implementation
- Do NOT assume project structure matches typical patterns

**PROHIBITED:**
- "I assume you have a src/solvers folder..." ❌
- "Typically this would be in..." ❌
- CORRECT: "Let me check your current structure..." ✅

---

## 3. PostgreSQL (Not Applicable for SiPhON)

**Note:** SiPhON does not use databases. Results are stored as:
- NumPy `.npz` files (simulation outputs)
- Jupyter notebooks (analysis + figures)
- CSV files (Monte Carlo results)

If data management becomes complex, revisit this section.

---

## Tool Priority Matrix

| Scenario | Primary Tool | Secondary Tool | Fallback |
|----------|--------------|----------------|----------|
| API usage question | Context7 | User-provided URL + read | Source code inspection |
| Code structure question | Filesystem | - | Ask user |
| Numerical method question | Context7 (library docs) | Academic references | Ask user |
| Library version conflict | Context7 (specific version) | Release notes URL | Ask user |

---

## Anti-Patterns to Avoid

### ❌ DON'T:
```
"In Eigen, you typically use SparseMatrix<double> for this..."
(Without checking Eigen 3.4 specific patterns)
```

### ✅ DO:
```
"Let me fetch Eigen 3.4 sparse matrix best practices first."
[calls context7: Eigen sparse matrix documentation]
```

### ❌ DON'T:
```
"Your solver probably needs a 5-point stencil discretization..."
```

### ✅ DO:
```
"Let me check your current FDE implementation first."
[calls filesystem: locate solver code, read implementation]
```

### ❌ DON'T:
```
"I'll add the mode solver to src/solvers/..."
```

### ✅ DO:
```
"Let me check your current project structure."
[calls filesystem: list directories, locate existing solver files]
```

---

## Zero-Tolerance Rules

1. **Never guess library APIs** - If you don't have Context7 docs, explicitly say "I need to fetch documentation for [X] before proceeding"

2. **Never assume numerical methods** - If implementing a solver, say "I need to verify the discretization scheme from [reference/docs]"

3. **Never fake file paths** - If you need to know where something is, say "Let me navigate your filesystem to locate [X]"

4. **When in doubt, fetch** - It's better to make 3 tool calls and be accurate than make 0 calls and be wrong

---

## Integration with SiPhON Workflow

Since SiPhON is numerically rigorous, tool usage is especially important:

- **Before implementing discretization schemes** → Context7: Eigen sparse matrix assembly
- **Before configuring eigensolvers** → Context7: Spectra solver parameters
- **Before Python binding changes** → Context7: pybind11 memory mapping
- **Before dependency updates** → Context7: check version compatibility for Eigen, Spectra

---

## Quick Reference

```
Library API unclear?        → context7 [library] [version]
Don't know file location?   → filesystem list/read
Numerical method unclear?   → context7 [library] + academic references
Documentation link broken?  → read [url]
```

---

## System Components

### 1. Analytical Ring Model (`siphon.ring`)

#### Purpose
Establish physics baseline using closed-form transfer functions.

#### Key Outputs
- **FSR (Free Spectral Range)**: λ² / (2πR·n_g)
- **Quality Factor (Q)**: Related to linewidth and losses
- **Extinction Ratio (ER)**: On-resonance vs. off-resonance transmission
- **Thermal Drift**: Δλ/ΔT via thermo-optic coefficient (dn/dT ≈ 1.8×10⁻⁴/K)

#### Parameters
- Radius (R)
- Coupling coefficient (κ)
- Waveguide loss (α)
- Group index (n_g)

---

### 2. C++ Mode Solver (`siphon.solver`)

#### Architecture
- **Discretization:** 2D scalar Helmholtz equation via 5-point finite-difference stencil
- **Sparse Matrix:** `Eigen::SparseMatrix<double>` for operator assembly
- **Eigensolver:** Spectra shift-and-invert Arnoldi (targets fundamental mode only)
- **Output:** Effective index (n_eff) and mode profile

#### Key Classes (Planned)
```cpp
class HelmholtzSolver {
    // Assemble sparse Laplacian + refractive index operator
    Eigen::SparseMatrix<double> AssembleOperator(const Grid2D& n_squared);

    // Solve for fundamental mode
    ModeResult SolveFundamental(double k0, double n_guess);
};

struct ModeResult {
    double n_eff;
    Eigen::VectorXd field;  // Electric field profile
};
```

#### Python Binding
```python
import siphon.solver as solver

# Pass NumPy array (zero-copy via pybind11)
n_grid = np.array(...)  # 2D refractive index profile
result = solver.solve_mode(n_grid, wavelength=1.55e-6)

print(f"n_eff = {result.n_eff}")
```

---

### 3. Variability Engine (`siphon.yield`)

#### Monte Carlo Sampling
- **Input Distributions:**
  - Waveguide width: w ~ N(500nm, 10nm)
  - Silicon thickness: h ~ N(220nm, 5nm)
- **Sensitivity Mapping:**
  - ∂n_eff/∂w via finite differences (solver sweeps)
  - ∂n_eff/∂h via finite differences
- **Thermal Power Calculation:**
  - Wavelength shift: Δλ = (∂λ/∂n_eff) × (∂n_eff/∂w) × Δw + ...
  - Heater power: P = Δλ / (thermo-optic tuning efficiency)

#### Yield Metric
**Definition:** Fraction of devices tunable to target wavelength without exceeding max heater power (e.g., 10mW).

---

### 4. Analysis & Visualization

#### Jupyter Notebooks
- `notebooks/01_analytical_baseline.ipynb`: Ring model validation
- `notebooks/02_solver_convergence.ipynb`: Grid refinement study
- `notebooks/03_sensitivity_maps.ipynb`: n_eff(w, h) surfaces
- `notebooks/04_yield_analysis.ipynb`: Monte Carlo results + power histograms

#### Key Plots
- FSR vs. radius
- Q vs. loss coefficient
- n_eff sensitivity heatmaps
- Heater power distribution (histogram)
- Yield vs. tolerance curves

---

## Build & Run

### Prerequisites
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Python 3.11+
- Eigen 3.4+
- Spectra library
- pybind11

### Quick Start
```bash
# Clone repository
git clone <repo-url> siphon
cd siphon

# Build C++ solver + Python bindings
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install Python package (editable mode)
cd ..
pip install -e .

# Run analytical baseline
python scripts/run_analytical.py

# Run mode solver (once implemented)
python scripts/run_solver.py

# Run Monte Carlo yield analysis (once implemented)
python scripts/run_yield_analysis.py

# Launch Jupyter for interactive analysis
jupyter notebook notebooks/
```

---

## Tuning Parameters

### Mode Solver Configuration
```cpp
// Grid resolution
int nx = 256;  // Points in x-direction
int ny = 256;  // Points in y-direction
double dx = 10e-9;  // 10nm grid spacing

// Eigenvalue solver
int n_modes = 1;           // Only fundamental mode
double shift = n_guess^2;  // Shift-invert around estimated n_eff
int max_iter = 1000;
double tol = 1e-10;
```

### Monte Carlo Configuration
```python
# Process variation
sigma_w = 10e-9  # 10nm width variation (3σ)
sigma_h = 5e-9   # 5nm thickness variation (3σ)

# Sampling
n_samples = 10000  # Monte Carlo sample size

# Thermal budget
max_heater_power = 10e-3  # 10mW limit
```

---

## Known Limitations & Future Work

### Current Scope (MVP)
- 2D scalar solver (TE-like modes only, no polarization coupling)
- Straight waveguides (no bend modeling)
- Single-ring geometry (no coupled resonators)
- Process variation only (no temperature variation)

### Roadmap (See [ROADMAP.md](ROADMAP.md))
- **0.1:** Analytical baseline + thermal constraints
- **0.2:** Variability engine + yield metric
- **0.3:** C++ solver core + Python bindings
- **0.4:** Integrated sensitivity analysis loop
- **1.0:** Full documentation, ADRs, reproducible builds

### Stretch Goals (Post-1.0)
- Semi-vectorial solver (dielectric discontinuity handling)
- Group index dispersion (∂n_eff/∂λ)
- Bent waveguide conformal mapping
- Multi-ring coupled systems

---

## Performance Expectations

### Mode Solver (Target)
- Grid size: 256×256
- Time per solve: <1 second (C++ implementation)
- Memory: <100MB per solve

### Monte Carlo (Target)
- 10,000 samples with full solver calls
- Total runtime: <3 hours (parallelized)
- Output: Power distribution histogram + yield percentage

---

## File Structure (Planned)

```
siphon/
├── CMakeLists.txt               # Top-level build
├── src/
│   ├── solver/
│   │   ├── helmholtz.cpp        # 2D FDE discretization
│   │   ├── grid.cpp             # Spatial grid utilities
│   │   └── mode.cpp             # Mode result structures
│   ├── bindings/
│   │   └── pybind_solver.cpp    # Python bindings
│   └── CMakeLists.txt
├── python/
│   ├── siphon/
│   │   ├── __init__.py
│   │   ├── ring.py              # Analytical ring model
│   │   ├── yield_analysis.py   # Monte Carlo engine
│   │   └── solver.py            # (Generated by pybind11)
│   └── setup.py
├── notebooks/
│   ├── 01_analytical_baseline.ipynb
│   ├── 02_solver_convergence.ipynb
│   ├── 03_sensitivity_maps.ipynb
│   └── 04_yield_analysis.ipynb
├── scripts/
│   ├── run_analytical.py
│   ├── run_solver.py
│   └── run_yield_analysis.py
├── tests/
│   ├── test_ring.py
│   ├── test_solver.cpp
│   └── test_yield.py
├── docs/
│   ├── architecture.md
│   ├── adr/                     # Architecture Decision Records
│   └── references/              # Academic papers, datasheets
├── ROADMAP.md
├── CLAUDE.md                    # This file
└── TODO.md
```

---

## Validation & Testing

### Unit Tests
- Analytical ring model vs. literature (FSR, Q formulas)
- Sparse matrix assembly (verify symmetry, sparsity pattern)
- Eigenvalue solver convergence (compare to known solutions)

### Convergence Studies
- Grid refinement: n_eff vs. 1/N (should show O(h²) for 5-point stencil)
- Eigenvalue solver: residual vs. iteration count

### Benchmarking
- C++ solver vs. pure NumPy/SciPy implementation (expect 10-100× speedup)
- Memory scaling: sparse matrix size vs. grid resolution

---

## Academic Rigor

### Documentation Requirements
- **Every assumption documented**: Scalar approximation, neglected effects
- **Every parameter justified**: Grid spacing, solver tolerance, MC sample size
- **Every result validated**: Convergence, comparison to literature

### Architecture Decision Records (ADRs)
To be written during implementation:
- ADR-001: Choice of scalar vs. vectorial formulation
- ADR-002: Shift-and-invert vs. direct eigenvalue solver
- ADR-003: pybind11 vs. Cython vs. raw Python/C API
- ADR-004: Yield metric definition (power budget vs. passive alignment)

---

## Contributing

Key principles for SiPhON development:

1. **Verify numerics first.** Grid convergence, solver residuals, analytical limits.
2. **Document assumptions.** What physics is included? What's neglected?
3. **Reproducibility.** Lock versions, seed RNGs, save full parameter sets.
4. **Performance is secondary.** Correctness > speed (but don't be wasteful).

See `.claude/rules/` for coding standards.

---

## License

(TBD - academic/open-source pending university IP review)

---

**Last Updated:** 2026-01-04
**Architecture Version:** SiPhON v0.1-dev (Pre-Implementation)
