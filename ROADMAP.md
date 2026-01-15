# SiPhON - Silicon Photonics with Open Numerics

## Goals

* **Architect** a high-performance hybrid simulation pipeline: C++17 numerics core, Python analysis layer.
* **Implement** a custom 2D Finite-Difference Eigenmode (FDE) solver to eliminate black-box dependencies.
* **Demonstrate** "Physics → Numerics → Yield" reasoning, specifically connecting process variation to thermal power budgets.
* **Deliver** a rigorous, reproducible artifact (CMake build, pybind11 interfaces, Jupyter reporting).

---

## Architecture & Stack

* **Core:** C++17, Eigen (Sparse Linear Algebra), Spectra (Sparse Eigenvalue Solver).
* **Interface:** `pybind11` / `nanobind` (Zero-copy memory mapping).
* **Analysis:** Python (NumPy, SciPy, Matplotlib), Jupyter.
* **Build:** CMake.

---

## Milestones

* **0.1 Core Physics Baseline** (Analytical models, FSR, Q, Thermal limits)
* **0.2 Variability & Yield Architecture** (Monte Carlo engine, Power Budget metric)
* **0.3 C++ Computational Core** (Sparse solver, Pybind11 integration)
* **0.4 The Hybrid Loop** (Zero-copy sweeps, sensitivity analysis)
* **1.0 Portfolio Release** (Docs, ADRs, Reproducible Build)

---

## 0.1 Core Physics Baseline

### Analytical Ring Model (P0)

* [ ] Implement analytical all-pass transfer function (`siphon.ring`).
* [ ] Parameterise geometry: Radius (), coupling (), loss ().
* [ ] Extract Key Metrics:
* Extinction Ratio (ER).
* Linewidth () & Quality Factor ().
* Free Spectral Range (FSR).



### Thermal Constraints (P0)

* [ ] Model thermo-optic effect in Silicon ().
* [ ] Calculate resonance drift ().
* [ ] Define **Heater Power Budget**: Power required to shift resonance by 1 FSR.

---

## 0.2 Variability & Yield Analysis

### Process Variation (P0)

* [ ] Define fabrication tolerance priors:
* Waveguide Width:  (e.g., 10nm).
* Silicon Thickness:  (e.g., 5nm).


* [ ] Map  (using analytical approximation first).

### Yield Definition (P0)

* [ ] **Metric Upgrade:** Shift from "Wavelength Hit" to **"Thermal Overhead."**
* *Yield = % of devices that can be tuned to target wavelength without exceeding max heater power.*


* [ ] Implement vectorized Monte Carlo sampler (avoid raw Python loops).

---

## 0.3 Custom EM Mode Solver (C++ Core)

### Numerical Core (P0)

* [ ] Setup C++17 CMake project structure.
* [ ] Implement 2D Scalar Helmholtz discretization (5-point stencil).
* [ ] **Matrix Assembly:** Construct `Eigen::SparseMatrix` for arbitrary  grids.
* [ ] **Solver:** Implement Shift-and-Invert Arnoldi (via `Spectra` or ARPACK) to find only the fundamental mode.

### Python Bindings (P0)

* [ ] Integrate `pybind11` to expose the `Solver` class to Python.
* [ ] Implement `numpy`  `Eigen` memory mapping (pass geometry grids without copy).
* [ ] Return `ModeProfile` and `n_eff` directly as Python objects.

### Validation (P0)

* [ ] Grid convergence study (convergence rate vs. ).
* [ ] Benchmarking: Compare execution time (Pure Python vs. C++ bind).

---

## 0.4 The Hybrid Loop (Solver-Device Integration)

### Sensitivity Extraction (P0)

* [ ] Replace analytical  with C++ Solver calls inside the Python loop.
* [ ] Generate  surfaces via high-speed solver sweeps.
* [ ] Calculate numerical sensitivities: .

### Full Yield Run (P0)

* [ ] Re-run Monte Carlo using Solver-derived sensitivities.
* [ ] Generate the "Heater Power Distribution" histogram.
* [ ] Quantify the cost of process control (e.g., "Improving width control by 5nm saves X mW heater power").

---

## 1.0 Portfolio Release

### Documentation & Narrative (P0)

* [ ] **Architecture Decision Records (ADRs):**
* Why C++ binding? (Performance vs. Complexity).
* Scalar vs. Vector limitations (explicitly acknowledge polarization inaccuracy).


* [ ] **Report:** "The Cost of Variance: A Numerically Rigorous SiPh Yield Study."

### Artifact Polish (P0)

* [ ] Single-command build (`pip install .` triggering CMake).
* [ ] Clean API documentation strings.
* [ ] Reproducible Jupyter Notebook (Results/Figures).

### Stretch Goals (P1)

* [ ] **Semi-Vectorial Solver:** Upgrade Laplacian to handle dielectric discontinuity (P1).
* [ ] **Group Index ():** Implement finite-difference dispersion calculation.
* [ ] **Bent Waveguides:** Add conformal mapping for small-radius bends.

---

## Decision Records

* [ ] **Interface:** `pybind11` selected over file I/O for Monte Carlo performance and type safety.
* [ ] **Solver:** Scalar approximation accepted for MVP; architecture supports Vectorial upgrade.
* [ ] **Yield:** Defined as "Power Budget Compliance" rather than "Passive Alignment."