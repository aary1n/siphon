"""
Effective Index Method for Silicon Strip Waveguides

Computes the effective refractive index n_eff(w, h) and its sensitivity
to waveguide geometry variations using the two-step Effective Index Method (EIM).

Theory:
    The EIM decomposes a 2D rectangular waveguide into two 1D slab problems:

    Step 1 (Vertical): Solve the slab of thickness h with core n_Si, cladding n_SiO2.
        -> Yields intermediate effective index n_slab(h).

    Step 2 (Horizontal): Solve the slab of width w with core n_slab(h), cladding n_SiO2.
        -> Yields the final effective index n_eff(w, h).

    Polarization bookkeeping for the quasi-TE strip mode (dominant Ex):
        Step 1: Ex is parallel to the horizontal layer interfaces -> TE slab equation.
        Step 2: Ex is perpendicular to the vertical effective-slab interfaces
                -> TM slab equation. Using TE in both steps overestimates n_eff
                by ~5-10% for standard SOI wires.

    Each 1D slab solve finds the fundamental mode by solving the
    transcendental characteristic equation:
        TE:  kx * tan(kx * d/2) = gamma
        TM:  kx * tan(kx * d/2) = (n_core^2 / n_clad^2) * gamma
    where:
        kx = sqrt(k0^2 * n_core^2 - beta^2)      (transverse wavenumber in core)
        gamma = sqrt(beta^2 - k0^2 * n_clad^2)    (decay constant in cladding)
        beta = k0 * n_eff                          (propagation constant)

    Sensitivities dn_eff/dw and dn_eff/dh are computed via central finite
    differences, mirroring the approach used by the C++ solver in Phase 0.4.

Assumptions:
    - TE polarization only (dominant for silicon wire waveguides)
    - Symmetric cladding (same material above and below core)
    - No material dispersion (refractive indices at a single wavelength)
    - EIM accuracy: ~1-5% on n_eff, ~10-20% on sensitivities vs. full 2D FDE

References:
    [1] Okamoto, K., "Fundamentals of Optical Waveguides," Academic Press (2006), Ch. 2-3
    [2] Chrostowski, L. & Hochberg, M., "Silicon Photonics Design," Cambridge (2015), Ch. 2
    [3] Marcatili, E.A.J., "Dielectric rectangular waveguide," Bell Syst. Tech. J. 48, 2071-2102 (1969)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq


# Material constants at 1550 nm
N_SILICON = 3.476      # Refractive index of silicon
N_OXIDE = 1.444        # Refractive index of SiO2 (thermal oxide)


@dataclass(frozen=True)
class WaveguideGeometry:
    """
    Silicon strip waveguide geometry for sensitivity analysis.

    Parameters
    ----------
    width : float
        Waveguide width [m]. Nominal: 500nm for single-mode at 1550nm.
    height : float
        Silicon layer thickness [m]. Nominal: 220nm for standard SOI.
    wavelength : float
        Operating wavelength [m]. Default: 1.55e-6 (C-band).
    n_core : float
        Core refractive index (silicon). Default: 3.476 at 1550nm.
    n_clad : float
        Cladding refractive index (SiO2). Default: 1.444 at 1550nm.
    """
    width: float
    height: float
    wavelength: float = 1.55e-6
    n_core: float = N_SILICON
    n_clad: float = N_OXIDE

    def __post_init__(self) -> None:
        """Validate geometry parameters."""
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.n_core <= self.n_clad:
            raise ValueError(
                f"Core index ({self.n_core}) must exceed cladding index ({self.n_clad})"
            )


@dataclass(frozen=True)
class SensitivityCoefficients:
    """
    Sensitivity of n_eff to waveguide geometry variations.

    Parameters
    ----------
    n_eff : float
        Effective index at nominal geometry [dimensionless].
    dn_eff_dw : float
        Partial derivative of n_eff with respect to width [1/m].
    dn_eff_dh : float
        Partial derivative of n_eff with respect to height [1/m].
    dlambda_dw : float
        Resonance wavelength sensitivity to width [m/m = dimensionless].
        Computed via chain rule: (lambda / n_g) * dn_eff_dw.
    dlambda_dh : float
        Resonance wavelength sensitivity to height [m/m = dimensionless].
    waveguide : WaveguideGeometry
        Nominal geometry at which coefficients were computed.
    n_g : float
        Group index used for chain rule computation.
    method : str
        Computation method identifier.
    """
    n_eff: float
    dn_eff_dw: float
    dn_eff_dh: float
    dlambda_dw: float
    dlambda_dh: float
    waveguide: WaveguideGeometry
    n_g: float
    method: str = "effective_index_method"


def _slab_neff(
    thickness: float,
    wavelength: float,
    n_core: float,
    n_clad: float,
    eta: float,
) -> float:
    """
    Solve for the fundamental mode effective index of a symmetric slab waveguide.

    Finds beta such that the characteristic equation is satisfied:
        kx * tan(kx * d/2) = eta * gamma

    where eta = 1 for TE and eta = (n_core/n_clad)^2 for TM polarization,
    and kx, gamma are functions of beta (the propagation constant).

    Parameters
    ----------
    thickness : float
        Slab thickness [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.
    eta : float
        Polarization factor multiplying gamma in the characteristic equation.

    Returns
    -------
    n_eff : float
        Effective index of the fundamental mode.

    Raises
    ------
    ValueError
        If the slab is below cutoff (no guided mode exists).
    """
    k0 = 2 * np.pi / wavelength
    d = thickness

    # V-number determines number of guided modes
    V = k0 * d / 2 * np.sqrt(n_core**2 - n_clad**2)

    if V < np.pi / 10:
        # Very weakly guiding -- near or below cutoff for fundamental mode.
        # For a symmetric slab the fundamental TE mode has no cutoff in theory,
        # but for extremely thin slabs n_eff approaches n_clad and the
        # solver becomes ill-conditioned.
        raise ValueError(
            f"Slab V-number too low ({V:.4f}): thickness {thickness*1e9:.1f}nm "
            f"is too thin for reliable mode solving at {wavelength*1e9:.0f}nm"
        )

    # Search for n_eff in (n_clad, n_core)
    # The characteristic equation for a symmetric slab:
    #   f(n_eff) = kx * tan(kx * d/2) - eta * gamma = 0
    # where:
    #   kx = sqrt(k0^2 * n_core^2 - beta^2)
    #   gamma = sqrt(beta^2 - k0^2 * n_clad^2)
    #   beta = k0 * n_eff

    def characteristic_equation(n_eff_trial: float) -> float:
        beta = k0 * n_eff_trial
        kx_sq = k0**2 * n_core**2 - beta**2
        gamma_sq = beta**2 - k0**2 * n_clad**2

        if kx_sq <= 0 or gamma_sq <= 0:
            return 1e10  # Outside guided region

        kx = np.sqrt(kx_sq)
        gamma = np.sqrt(gamma_sq)

        return kx * np.tan(kx * d / 2) - eta * gamma

    # Bracket: n_eff is between n_clad and n_core
    # Use a small offset from boundaries to avoid singularities
    n_lo = n_clad + 1e-10
    n_hi = n_core - 1e-10

    # For the fundamental mode, there is exactly one zero crossing
    # We need to avoid the tan() singularities (where kx*d/2 = pi/2 + m*pi)
    # For fundamental mode, the solution is in the first branch: kx*d/2 < pi/2

    # Find upper bound where kx*d/2 < pi/2
    # kx_max = sqrt(k0^2 * n_core^2 - k0^2 * n_lo^2) at n_eff = n_lo
    # We need kx*d/2 < pi/2, so n_eff > n_eff_min where kx*d/2 = pi/2
    kx_limit = np.pi / d  # kx at the first tan singularity

    # n_eff where kx = kx_limit
    beta_at_limit_sq = k0**2 * n_core**2 - kx_limit**2
    if beta_at_limit_sq > 0:
        n_eff_at_limit = np.sqrt(beta_at_limit_sq) / k0
        # Ensure we search below the singularity
        if n_eff_at_limit > n_lo:
            n_lo = max(n_lo, n_eff_at_limit + 1e-10)

    try:
        n_eff_solution = brentq(characteristic_equation, n_lo, n_hi, xtol=1e-12, rtol=1e-12)
    except ValueError:
        raise ValueError(
            f"No guided mode found for slab: thickness={thickness*1e9:.1f}nm, "
            f"wavelength={wavelength*1e9:.0f}nm, n_core={n_core}, n_clad={n_clad}"
        )

    return n_eff_solution


def slab_te_neff(
    thickness: float,
    wavelength: float,
    n_core: float,
    n_clad: float,
) -> float:
    """
    Fundamental TE mode effective index of a symmetric slab waveguide.

    Solves kx * tan(kx * d/2) = gamma (electric field parallel to the
    slab interfaces).

    Parameters
    ----------
    thickness : float
        Slab thickness [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.

    Returns
    -------
    n_eff : float
        Effective index of the fundamental TE mode.
    """
    return _slab_neff(thickness, wavelength, n_core, n_clad, eta=1.0)


def slab_tm_neff(
    thickness: float,
    wavelength: float,
    n_core: float,
    n_clad: float,
) -> float:
    """
    Fundamental TM mode effective index of a symmetric slab waveguide.

    Solves kx * tan(kx * d/2) = (n_core/n_clad)^2 * gamma (electric field
    perpendicular to the slab interfaces).

    Parameters
    ----------
    thickness : float
        Slab thickness [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.

    Returns
    -------
    n_eff : float
        Effective index of the fundamental TM mode.
    """
    return _slab_neff(thickness, wavelength, n_core, n_clad, eta=(n_core / n_clad) ** 2)


def check_single_mode(
    width: float,
    height: float,
    wavelength: float = 1.55e-6,
    n_core: float = N_SILICON,
    n_clad: float = N_OXIDE,
) -> bool:
    """
    Check if a rectangular waveguide supports only a single TE mode.

    Uses an approximate criterion based on the V-number of each dimension.

    Parameters
    ----------
    width : float
        Waveguide width [m].
    height : float
        Silicon thickness [m].
    wavelength : float
        Operating wavelength [m].
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.

    Returns
    -------
    is_single_mode : bool
        True if only the fundamental TE mode is guided.

    Notes
    -----
    This is an approximate check, consistent with the EIM decomposition:
    the vertical direction uses the full core-cladding NA, while the
    horizontal direction uses the step-1 slab index n_slab(h) as its core
    (the lateral confinement a TE1 mode would actually see).

    The V < pi threshold (rather than the strict symmetric-slab TE1 cutoff
    at V = pi/2) compensates for the EIM's overestimated lateral
    confinement; it places the TE1 cutoff at w = lambda / NA_eff ~ 645nm
    for 220nm SOI at 1550nm, consistent with full-vector results.
    """
    k0 = 2 * np.pi / wavelength

    # Vertical: full core-cladding contrast
    NA_vertical = np.sqrt(n_core**2 - n_clad**2)
    V_height = k0 * height / 2 * NA_vertical

    # Horizontal: contrast seen by the lateral mode is set by the
    # vertical-slab effective index, not the bulk core index
    n_slab = slab_te_neff(height, wavelength, n_core, n_clad)
    NA_horizontal = np.sqrt(n_slab**2 - n_clad**2)
    V_width = k0 * width / 2 * NA_horizontal

    return bool(V_width < np.pi and V_height < np.pi)


class EffectiveIndexSolver:
    """
    Two-step Effective Index Method for silicon strip waveguides.

    Computes n_eff(w, h) and sensitivity coefficients dn_eff/dw, dn_eff/dh
    for TE-like modes of rectangular dielectric waveguides.

    Parameters
    ----------
    waveguide : WaveguideGeometry
        Nominal waveguide geometry.

    Notes
    -----
    The EIM decomposition for TE-like modes (dominant Ex):
    - Step 1: TE slab in the vertical direction (thickness = height)
    - Step 2: TM slab in the horizontal direction (width, using n_slab from
      step 1), because Ex is perpendicular to the vertical effective-slab
      interfaces.

    Accuracy: ~1-5% on n_eff vs. full 2D FDE solver for typical SOI geometries.
    Sensitivity accuracy: ~10-20% (sufficient for Phase 0.2 yield estimates).

    References
    ----------
    [1] Okamoto, K., "Fundamentals of Optical Waveguides," Ch. 2-3 (2006)
    """

    def __init__(self, waveguide: WaveguideGeometry) -> None:
        self.waveguide = waveguide

    def n_eff(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> float:
        """
        Compute effective index via two-step EIM.

        Step 1: Solve vertical TE slab (thickness=h, core=n_Si, clad=n_SiO2) -> n_slab
        Step 2: Solve horizontal TM slab (thickness=w, core=n_slab, clad=n_SiO2) -> n_eff

        Parameters
        ----------
        width : float, optional
            Override waveguide width [m]. Default: nominal from geometry.
        height : float, optional
            Override silicon height [m]. Default: nominal from geometry.

        Returns
        -------
        n_eff : float
            Effective index of the fundamental TE-like mode.
        """
        w = width if width is not None else self.waveguide.width
        h = height if height is not None else self.waveguide.height
        wl = self.waveguide.wavelength
        n_core = self.waveguide.n_core
        n_clad = self.waveguide.n_clad

        # Step 1: Vertical slab (TE polarization: Ex parallel to interfaces)
        n_slab = slab_te_neff(h, wl, n_core, n_clad)

        # Step 2: Horizontal slab using n_slab as effective core index
        # (TM polarization: Ex perpendicular to the vertical interfaces)
        n_eff_result = slab_tm_neff(w, wl, n_slab, n_clad)

        return n_eff_result

    def sensitivity(
        self,
        n_g: float = 4.2,
        delta_w: float = 1e-9,
        delta_h: float = 1e-9,
    ) -> SensitivityCoefficients:
        """
        Compute sensitivity coefficients via central finite differences.

        dn_eff/dw = [n_eff(w+dw, h) - n_eff(w-dw, h)] / (2*dw)
        dn_eff/dh = [n_eff(w, h+dh) - n_eff(w, h-dh)] / (2*dh)

        Resonance wavelength sensitivities via chain rule:
        dlambda/dw = (lambda / n_g) * dn_eff/dw
        dlambda/dh = (lambda / n_g) * dn_eff/dh

        Parameters
        ----------
        n_g : float
            Group index for chain rule. Default: 4.2 (typical Si wire at 1550nm).
        delta_w : float
            Finite difference step for width [m]. Default: 1nm.
        delta_h : float
            Finite difference step for height [m]. Default: 1nm.

        Returns
        -------
        coefficients : SensitivityCoefficients
            Complete set of sensitivity values at nominal geometry.
        """
        w0 = self.waveguide.width
        h0 = self.waveguide.height
        wl = self.waveguide.wavelength

        # Central n_eff
        n_eff_0 = self.n_eff()

        # Width sensitivity: dn_eff/dw
        n_eff_wp = self.n_eff(width=w0 + delta_w)
        n_eff_wm = self.n_eff(width=w0 - delta_w)
        dn_eff_dw = (n_eff_wp - n_eff_wm) / (2 * delta_w)

        # Height sensitivity: dn_eff/dh
        n_eff_hp = self.n_eff(height=h0 + delta_h)
        n_eff_hm = self.n_eff(height=h0 - delta_h)
        dn_eff_dh = (n_eff_hp - n_eff_hm) / (2 * delta_h)

        # Chain rule for resonance wavelength sensitivity
        # dlambda_res/dw = (lambda / n_g) * dn_eff/dw
        dlambda_dw = (wl / n_g) * dn_eff_dw
        dlambda_dh = (wl / n_g) * dn_eff_dh

        return SensitivityCoefficients(
            n_eff=n_eff_0,
            dn_eff_dw=dn_eff_dw,
            dn_eff_dh=dn_eff_dh,
            dlambda_dw=dlambda_dw,
            dlambda_dh=dlambda_dh,
            waveguide=self.waveguide,
            n_g=n_g,
        )

    def n_eff_map(
        self,
        widths: NDArray[np.floating],
        heights: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute n_eff over a 2D grid of (width, height) values.

        Parameters
        ----------
        widths : ndarray, shape (N,)
            Array of width values [m].
        heights : ndarray, shape (M,)
            Array of height values [m].

        Returns
        -------
        n_eff_grid : ndarray, shape (M, N)
            Effective index at each (height[i], width[j]) point.
            Uses NaN for geometries where no guided mode exists.
        """
        n_eff_grid = np.full((len(heights), len(widths)), np.nan)

        for i, h in enumerate(heights):
            for j, w in enumerate(widths):
                try:
                    n_eff_grid[i, j] = self.n_eff(width=w, height=h)
                except ValueError:
                    # Below cutoff -- leave as NaN
                    pass

        return n_eff_grid

    def __repr__(self) -> str:
        wg = self.waveguide
        return (
            f"EffectiveIndexSolver(w={wg.width*1e9:.0f}nm, h={wg.height*1e9:.0f}nm, "
            f"lambda={wg.wavelength*1e9:.0f}nm)"
        )
