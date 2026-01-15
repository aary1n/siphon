### SiPhON - Silicon Photonics with Open Numerics

**Goal:** Demonstrate "Physics → Numerics → Yield" reasoning for silicon photonic ring resonators.
**Artifact:** Rigorous portfolio piece with custom C++ solver, Python analysis, reproducible builds.

---

## Phase 0.1: Core Physics Baseline (Analytical Models)
*Establish theoretical foundation using closed-form equations*

### Analytical Ring Model (P0)
- [ ] **Ring Transfer Function:** Implement all-pass analytical model (`siphon.ring`).
    - [ ] Define geometry parameters: Radius (R), coupling coefficient (κ), waveguide loss (α).
    - [ ] Calculate transmission spectrum: T(λ) for given parameters.
    - [ ] Extract key metrics:
        - [ ] Free Spectral Range (FSR = λ² / (2πR·n_g))
        - [ ] Quality Factor (Q = λ / Δλ)
        - [ ] Extinction Ratio (ER = T_off / T_on)

### Thermal Constraints (P0)
- [ ] **Thermo-Optic Model:** Implement resonance wavelength shift.
    - [ ] Define silicon thermo-optic coefficient (dn/dT ≈ 1.8×10⁻⁴ K⁻¹).
    - [ ] Calculate resonance drift: Δλ_res / ΔT.
    - [ ] Define **Heater Power Budget** metric:
        - [ ] Power required to shift resonance by 1 FSR.
        - [ ] Map temperature change to wavelength shift to heater power.

### Validation (P0)
- [ ] **Literature Comparison:** Verify FSR, Q, ER against published ring resonator formulas.
- [ ] **Sanity Checks:** Ensure FSR decreases with increasing radius, Q increases with decreasing loss.

---

## Phase 0.2: Variability & Yield Architecture
*Monte Carlo engine connecting process variation to thermal overhead*

### Process Variation Model (P0)
- [ ] **Fabrication Tolerance Priors:** Define Gaussian distributions.
    - [ ] Waveguide width: w ~ N(500nm, σ_w) where σ_w = 10nm (example).
    - [ ] Silicon thickness: h ~ N(220nm, σ_h) where σ_h = 5nm (example).
    - [ ] Document source of tolerances (foundry specs, literature).

### Sensitivity Mapping (P0 - Analytical Approximation)
- [ ] **Analytical Gradient:** Calculate ∂n_eff/∂w using perturbation theory or empirical fits.
    - [ ] Use simplified slab waveguide approximation initially.
    - [ ] Document assumptions and expected error.
- [ ] **Wavelength Shift:** Map geometry variation to effective index to resonance wavelength.
    - [ ] Δλ_res = f(Δw, Δh) via chain rule.

### Yield Metric Definition (P0)
- [ ] **Thermal Overhead Yield:** Define as percentage of devices tunable within power budget.
    - [ ] Yield = % devices with required heater power < P_max (e.g., 10mW).
    - [ ] NOT just "wavelength hit" - explicitly account for tuning cost.

### Monte Carlo Engine (P0)
- [ ] **Vectorized Sampling:** Implement efficient Monte Carlo without raw Python loops.
    - [ ] Generate N samples: (w_i, h_i) ~ multivariate normal.
    - [ ] Compute wavelength shift distribution: Δλ_i for each sample.
    - [ ] Compute required heater power distribution: P_i for each sample.
    - [ ] Calculate yield: fraction with P_i < P_max.
- [ ] **Visualization:**
    - [ ] Histogram of heater power distribution.
    - [ ] Yield vs. tolerance curve (sweep σ_w, σ_h).

---

## Phase 0.3: Custom EM Mode Solver (C++ Core)
*Numerical rigor via custom 2D FDE implementation*

### Project Setup (P0)
- [ ] **CMake Structure:** Create modern CMake project.
    - [ ] Top-level `CMakeLists.txt` with C++17 standard.
    - [ ] Separate `src/solver/` and `src/bindings/` directories.
    - [ ] External dependencies: Eigen, Spectra (via git submodules or FetchContent).

### Helmholtz Solver Core (P0)
- [ ] **Grid Setup:** Implement 2D spatial grid (`Grid2D` class).
    - [ ] Store nx, ny, dx, dy parameters.
    - [ ] Map (i, j) → linear index for sparse matrix assembly.
- [ ] **Operator Assembly:** Build sparse Helmholtz operator.
    - [ ] 5-point finite-difference stencil for Laplacian (∇²).
    - [ ] Incorporate refractive index: ∇²E + k₀²n²(x,y)E = β²E.
    - [ ] Return `Eigen::SparseMatrix<double>` in COO or CSR format.
- [ ] **Boundary Conditions:** Implement PML or simple Dirichlet BCs.
    - [ ] Document choice and limitations.

### Eigenvalue Solver Integration (P0)
- [ ] **Spectra Configuration:** Setup shift-and-invert Arnoldi solver.
    - [ ] Target only fundamental mode (n_modes = 1).
    - [ ] Shift parameter: σ = (k₀ · n_guess)² for faster convergence.
    - [ ] Convergence tolerance: tol = 1e-10 (or document choice).
- [ ] **Mode Extraction:**
    - [ ] Extract eigenvalue β² → n_eff = β / k₀.
    - [ ] Extract eigenvector (electric field profile).

### Python Bindings (P0)
- [ ] **pybind11 Integration:** Expose C++ solver to Python.
    - [ ] Bind `HelmholtzSolver` class with `solve_mode()` method.
    - [ ] Zero-copy NumPy ↔ Eigen interface:
        - [ ] Input: NumPy array (2D n² profile) → Eigen::MatrixXd (map, no copy).
        - [ ] Output: Eigen::VectorXd (field) → NumPy array (map, no copy).
    - [ ] Return `ModeResult` object with `n_eff` and `field` attributes.
- [ ] **Build Integration:** CMake builds Python extension module.
    - [ ] Use `pybind11_add_module()` CMake function.
    - [ ] Install to `python/siphon/solver.so` (or `.pyd` on Windows).

### Validation (P0)
- [ ] **Analytical Comparison:** Test against known solutions.
    - [ ] Slab waveguide (1D reduction, exact solution available).
    - [ ] Homogeneous medium (should match plane wave k₀·n).
- [ ] **Grid Convergence Study:** Verify O(h²) convergence for 5-point stencil.
    - [ ] Solve at multiple resolutions: 64², 128², 256².
    - [ ] Plot log(error) vs. log(dx) → slope should be ~2.
- [ ] **Benchmarking:** Compare C++ solver vs. pure NumPy/SciPy implementation.
    - [ ] Measure wall-clock time for 256×256 grid.
    - [ ] Expect 10-100× speedup for C++.

---

## Phase 0.4: The Hybrid Loop (Solver-Device Integration)
*Close the loop: solver → sensitivity → yield*

### Sensitivity Extraction (P0)
- [ ] **Replace Analytical ∂n_eff/∂w:** Use C++ solver for true numerical derivatives.
    - [ ] Perturb geometry: solve n_eff(w), n_eff(w + δw), n_eff(w - δw).
    - [ ] Finite difference: ∂n_eff/∂w ≈ [n_eff(w+δw) - n_eff(w-δw)] / (2δw).
    - [ ] Repeat for thickness: ∂n_eff/∂h.
- [ ] **Sensitivity Surface:** Generate n_eff(w, h) via 2D parameter sweep.
    - [ ] Grid of (w, h) points → call solver → store n_eff.
    - [ ] Visualize as heatmap/contour plot.

### Full Yield Run (P0)
- [ ] **Solver-Based Monte Carlo:** Re-run MC using numerical sensitivities.
    - [ ] For each sample (w_i, h_i), compute Δn_eff from sensitivity map.
    - [ ] Map Δn_eff → Δλ_res → heater power.
    - [ ] Generate heater power distribution histogram.
- [ ] **Yield Quantification:** Calculate percentage within power budget.
    - [ ] Compare to analytical approximation (Phase 0.2).
    - [ ] Quantify error introduced by analytical assumptions.

### Analysis Deliverable (P0)
- [ ] **Jupyter Notebook:** `04_yield_analysis.ipynb`
    - [ ] Load MC results, plot power distribution.
    - [ ] Sensitivity analysis: "Improving width control by 5nm saves X mW heater power."
    - [ ] Yield vs. tolerance curves (sweep σ_w with solver-based sensitivities).

---

## Phase 1.0: Portfolio Release (Polish & Documentation)
*Transform working code into rigorous artifact*

### Documentation & Narrative (P0)
- [ ] **Architecture Decision Records (ADRs):**
    - [ ] ADR-001: Scalar vs. Vectorial FDE (justify scalar approximation, error bounds).
    - [ ] ADR-002: Shift-and-invert vs. direct eigensolvers (performance trade-offs).
    - [ ] ADR-003: pybind11 vs. Cython (why zero-copy matters for this use case).
    - [ ] ADR-004: Yield metric definition (thermal budget vs. passive alignment).
- [ ] **Technical Report:** "The Cost of Variance: A Numerically Rigorous SiPh Yield Study"
    - [ ] Introduction: motivation, problem statement.
    - [ ] Methodology: analytical model, FDE formulation, MC sampling.
    - [ ] Results: sensitivity maps, yield curves, power distributions.
    - [ ] Discussion: assumptions, limitations, future work.

### Artifact Polish (P0)
- [ ] **Single-Command Build:** Implement `pip install .` triggering CMake.
    - [ ] Use `setup.py` with `cmake_build_ext` or `scikit-build`.
    - [ ] Ensure dependencies (Eigen, Spectra) are fetched automatically.
- [ ] **API Documentation:** Docstrings for all public functions.
    - [ ] Python: NumPy-style docstrings with parameters, returns, examples.
    - [ ] C++: Doxygen-compatible comments.
- [ ] **Reproducible Notebooks:** All Jupyter notebooks re-runnable from scratch.
    - [ ] Clear outputs, no manual steps.
    - [ ] Document runtime (e.g., "MC takes ~30 min on 4-core laptop").

### Testing (P0)
- [ ] **Unit Tests:**
    - [ ] Python: `pytest` for analytical ring model, MC engine.
    - [ ] C++: Google Test or Catch2 for solver core.
- [ ] **Integration Test:** Full pipeline from geometry → n_eff → yield.
    - [ ] Smoke test: ensure solver runs, produces reasonable n_eff.
    - [ ] Regression test: compare output to known-good reference.

### Continuous Integration (P1 - Optional)
- [ ] **GitHub Actions:** Automate build + test on push.
    - [ ] Linux (GCC, Clang), Windows (MSVC), macOS.
    - [ ] Cache CMake dependencies for speed.

---

## Stretch Goals (P1 - Post-1.0)

### Semi-Vectorial Solver (P1)
- [ ] **Upgrade Discretization:** Handle ε(x, y) discontinuities properly.
    - [ ] Yee grid for field components.
    - [ ] Interface averaging for permittivity.
- [ ] **Validation:** Compare to full vectorial MODE solver (if available).

### Group Index Calculation (P1)
- [ ] **Dispersion:** Compute ∂n_eff/∂λ via finite differences.
    - [ ] Solve at λ₀ ± δλ, extract n_g = n_eff - λ(∂n_eff/∂λ).
- [ ] **Impact on FSR:** Recalculate FSR using n_g instead of n_eff approximation.

### Bent Waveguide Support (P1)
- [ ] **Conformal Mapping:** Transform bent waveguide to equivalent straight guide.
    - [ ] Implement coordinate transformation in grid setup.
    - [ ] Validate against analytical bend loss formulas (if available).

### Multi-Ring Coupled Systems (P1)
- [ ] **Coupled-Mode Theory:** Extend analytical model to 2+ rings.
- [ ] **Solver Generalization:** Handle multiple waveguide regions in single simulation.

---

## Milestones

- [ ] **M0.1 (Analytical Baseline):** Ring model + thermal constraints working. Plots match literature.
- [ ] **M0.2 (Variability Engine):** Monte Carlo with analytical gradients. Yield histogram generated.
- [ ] **M0.3 (C++ Core):** Helmholtz solver compiles, passes convergence test, Python binding works.
- [ ] **M0.4 (Hybrid Loop):** Solver-based sensitivities integrated into MC. Final yield numbers produced.
- [ ] **M1.0 (Release):** Documentation complete, build automated, notebooks reproducible.

---

## Acceptance Criteria

### Phase 0.1
- [ ] Analytical FSR matches λ² / (2πR·n_g) formula within floating-point error.
- [ ] Thermal model predicts Δλ/ΔT ≈ 0.1 nm/K (typical silicon ring).

### Phase 0.2
- [ ] Monte Carlo runs 10,000 samples in <10 seconds (vectorized NumPy).
- [ ] Yield histogram shows expected normal-ish distribution of heater powers.

### Phase 0.3
- [ ] C++ solver grid convergence shows O(h²) scaling.
- [ ] n_eff error vs. analytical slab waveguide solution < 1%.
- [ ] Python binding returns identical results to C++ unit test.

### Phase 0.4
- [ ] Solver-based yield differs from analytical by <10% (validates analytical assumptions).
- [ ] Sensitivity map (n_eff vs. w, h) is smooth and physically reasonable.

### Phase 1.0
- [ ] `pip install .` succeeds on fresh Ubuntu/Windows/macOS environment.
- [ ] All notebooks execute without errors in <1 hour total.
- [ ] Technical report is >5 pages with figures, references, ADRs.

---

## Current Status

**Last Updated:** 2026-01-04
**Current Phase:** Pre-0.1 (Architecture & Planning)
**Blockers:** None (ready to start implementation)

---

## Notes

- **Performance is secondary to correctness.** Validate first, optimize later.
- **Document every assumption.** Scalar approximation error? Grid resolution trade-off? Write it down.
- **Version lock everything.** Eigen 3.4.0, Python 3.11.x, etc. Reproducibility requires it.
- **Use academic rigor.** This is a portfolio piece - treat it like a journal submission.
