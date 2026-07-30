"""
Benchmark: C++ FDE solver (Spectra) vs pure Python (scipy.sparse eigsh).

Both paths solve the identical eigenproblem: the same 5-point Helmholtz
operator for a 500x220nm silicon strip waveguide, shift-and-invert around
the same target. The scipy path assembles with scipy.sparse and solves
with ARPACK; the C++ path assembles and solves entirely in the extension
(one solve_mode call), so the comparison includes assembly for both.

Run:  python scripts/benchmark_solver.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from siphon import solver
from siphon.fde import build_rect_waveguide_n
from siphon.sensitivity import N_SILICON

WAVELENGTH = 1.55e-6
K0 = 2 * np.pi / WAVELENGTH


def assemble_scipy(n_grid: np.ndarray, dx: float, dy: float) -> sp.csc_matrix:
    ny, nx = n_grid.shape
    N = nx * ny
    main = (-2 / dx**2 - 2 / dy**2) + K0**2 * (n_grid.ravel() ** 2)
    ex = np.ones(N - 1)
    ex[nx - 1 :: nx] = 0
    ey = np.ones(N - nx)
    H = (
        sp.diags(main)
        + sp.diags(ex / dx**2, 1)
        + sp.diags(ex / dx**2, -1)
        + sp.diags(ey / dy**2, nx)
        + sp.diags(ey / dy**2, -nx)
    )
    return H.tocsc()


def bench(h: float, repeats: int = 3) -> None:
    n_grid = build_rect_waveguide_n(
        500e-9, 220e-9, domain_x=1.9e-6, domain_y=1.62e-6, dx=h, dy=h
    )
    ny, nx = n_grid.shape

    # C++ path (assembly + Spectra shift-invert), best of `repeats`
    t_cpp = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = solver.solve_mode(n_grid, WAVELENGTH, h, h)
        t_cpp.append(time.perf_counter() - t0)

    # Python path (scipy assembly + ARPACK shift-invert)
    sigma = (K0 * N_SILICON * 1.0001) ** 2
    t_py = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        H = assemble_scipy(n_grid, h, h)
        vals = spla.eigsh(H, k=1, sigma=sigma, which="LM", return_eigenvectors=False)
        t_py.append(time.perf_counter() - t0)

    n_eff_py = np.sqrt(vals[0]) / K0
    agree = abs(r.n_eff - n_eff_py) / n_eff_py
    print(
        f"{nx:>4} x {ny:<4} ({nx*ny:>7,d} unknowns)  "
        f"C++ {min(t_cpp)*1e3:9.1f} ms   scipy {min(t_py)*1e3:9.1f} ms   "
        f"speedup {min(t_py)/min(t_cpp):5.1f}x   |dn_eff| = {agree:.1e}"
    )


if __name__ == "__main__":
    print(f"Benchmark: 500x220nm Si strip @ {WAVELENGTH*1e9:.0f}nm, "
          f"fundamental mode, tol=1e-10\n")
    print(f"{'grid':>4}   {'':>18}  {'C++ (Spectra)':>13}   {'scipy (ARPACK)':>15}")
    for h in [20e-9, 10e-9, 5e-9]:
        bench(h)
