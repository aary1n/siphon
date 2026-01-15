"""
Analytical Ring Resonator Model

Implements the all-pass transfer function for silicon photonic ring resonators.
This module provides the physics baseline for Phase 0.1, using closed-form
equations to calculate transmission spectra and extract key metrics.

Theory:
    The all-pass ring resonator transfer function is:

        T(φ) = |E_out / E_in|² = (a² - 2ra·cos(φ) + r²) / (1 - 2ra·cos(φ) + (ra)²)

    where:
        - a = exp(-α·L/2) is the round-trip amplitude transmission (loss)
        - r = sqrt(1 - κ²) is the self-coupling coefficient
        - κ is the cross-coupling coefficient (power coupling)
        - φ = β·L = (2π·n_eff/λ)·2πR is the round-trip phase
        - L = 2πR is the ring circumference

References:
    [1] Bogaerts, W. et al., "Silicon microring resonators," Laser Photon. Rev. 6, 47-73 (2012)
    [2] Yariv, A., "Universal relations for coupling of optical power," Electron. Lett. 36, 321-322 (2000)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional


# Physical constants
C_VACUUM = 299792458.0  # Speed of light in vacuum [m/s]


@dataclass(frozen=True)
class RingGeometry:
    """
    Ring resonator geometry parameters.

    Parameters
    ----------
    radius : float
        Ring radius [m]. Typical: 5-50 μm.
    kappa : float
        Power coupling coefficient (0 < κ < 1). Related to gap distance.
        Typical: 0.1-0.5 for add-drop, 0.05-0.2 for high-Q.
    alpha : float
        Waveguide propagation loss [dB/cm]. Typical: 1-3 dB/cm for silicon.
    n_eff : float
        Effective refractive index. Typical: ~2.4 for TE mode in Si wire.
    n_g : float
        Group index (for FSR calculation). Typical: ~4.2 for Si wire at 1550nm.
    """
    radius: float       # [m]
    kappa: float        # Power coupling coefficient [dimensionless]
    alpha: float        # Propagation loss [dB/cm]
    n_eff: float        # Effective index [dimensionless]
    n_g: float          # Group index [dimensionless]

    def __post_init__(self) -> None:
        """Validate geometry parameters."""
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
        if not 0 < self.kappa < 1:
            raise ValueError(f"Coupling coefficient κ must be in (0, 1), got {self.kappa}")
        if self.alpha < 0:
            raise ValueError(f"Loss coefficient must be non-negative, got {self.alpha}")
        if self.n_eff <= 0:
            raise ValueError(f"Effective index must be positive, got {self.n_eff}")
        if self.n_g <= 0:
            raise ValueError(f"Group index must be positive, got {self.n_g}")

    @property
    def circumference(self) -> float:
        """Ring circumference L = 2πR [m]."""
        return 2 * np.pi * self.radius

    @property
    def self_coupling(self) -> float:
        """Self-coupling coefficient r = sqrt(1 - κ²) [dimensionless]."""
        return np.sqrt(1 - self.kappa**2)

    @property
    def round_trip_loss(self) -> float:
        """
        Round-trip amplitude transmission a = exp(-α·L/2) [dimensionless].

        Note: α is converted from dB/cm to Np/m internally.
        """
        # Convert dB/cm to Np/m: α [Np/m] = α [dB/cm] × 100 × ln(10)/20
        alpha_np_per_m = self.alpha * 100 * np.log(10) / 20
        return np.exp(-alpha_np_per_m * self.circumference / 2)


@dataclass(frozen=True)
class RingMetrics:
    """
    Extracted ring resonator metrics.

    Parameters
    ----------
    fsr : float
        Free Spectral Range [m].
    fsr_ghz : float
        Free Spectral Range [GHz].
    linewidth : float
        3dB linewidth Δλ [m].
    quality_factor : float
        Loaded quality factor Q = λ/Δλ.
    extinction_ratio_db : float
        Extinction ratio (on/off ratio) [dB].
    finesse : float
        Finesse F = FSR/Δλ.
    resonance_wavelength : float
        Reference resonance wavelength [m].
    """
    fsr: float
    fsr_ghz: float
    linewidth: float
    quality_factor: float
    extinction_ratio_db: float
    finesse: float
    resonance_wavelength: float


class RingResonator:
    """
    All-pass silicon ring resonator model.

    Calculates the analytical transfer function and extracts key metrics
    (FSR, Q, ER, finesse) for a given ring geometry.

    Parameters
    ----------
    geometry : RingGeometry
        Ring resonator geometry and material parameters.

    Examples
    --------
    >>> geom = RingGeometry(
    ...     radius=10e-6,      # 10 μm radius
    ...     kappa=0.2,         # 20% power coupling
    ...     alpha=2.0,         # 2 dB/cm loss
    ...     n_eff=2.4,         # Effective index
    ...     n_g=4.2            # Group index
    ... )
    >>> ring = RingResonator(geom)
    >>> metrics = ring.metrics(wavelength=1.55e-6)
    >>> print(f"FSR = {metrics.fsr * 1e9:.2f} nm")
    >>> print(f"Q = {metrics.quality_factor:.0f}")
    """

    def __init__(self, geometry: RingGeometry) -> None:
        self.geometry = geometry

    def round_trip_phase(self, wavelength: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """
        Calculate round-trip phase φ = β·L = (2π·n_eff/λ)·L.

        Parameters
        ----------
        wavelength : array_like
            Wavelength [m].

        Returns
        -------
        phi : ndarray
            Round-trip phase [rad].
        """
        wavelength = np.atleast_1d(wavelength)
        beta = 2 * np.pi * self.geometry.n_eff / wavelength
        return beta * self.geometry.circumference

    def transmission(self, wavelength: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """
        Calculate power transmission T(λ) through the all-pass ring.

        The all-pass transfer function is:
            T = (a² - 2ra·cos(φ) + r²) / (1 - 2ra·cos(φ) + (ra)²)

        Parameters
        ----------
        wavelength : array_like
            Wavelength [m].

        Returns
        -------
        T : ndarray
            Power transmission (0 to 1).
        """
        wavelength = np.atleast_1d(wavelength)
        phi = self.round_trip_phase(wavelength)

        a = self.geometry.round_trip_loss
        r = self.geometry.self_coupling

        cos_phi = np.cos(phi)

        numerator = a**2 - 2 * r * a * cos_phi + r**2
        denominator = 1 - 2 * r * a * cos_phi + (r * a)**2

        return numerator / denominator

    def transmission_db(self, wavelength: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """
        Calculate power transmission in dB.

        Parameters
        ----------
        wavelength : array_like
            Wavelength [m].

        Returns
        -------
        T_dB : ndarray
            Power transmission [dB].
        """
        T = self.transmission(wavelength)
        return 10 * np.log10(T)

    def fsr(self, wavelength: float) -> float:
        """
        Calculate Free Spectral Range.

        FSR = λ² / (n_g · L)

        Parameters
        ----------
        wavelength : float
            Center wavelength [m].

        Returns
        -------
        fsr : float
            Free Spectral Range [m].
        """
        return wavelength**2 / (self.geometry.n_g * self.geometry.circumference)

    def fsr_frequency(self, wavelength: float) -> float:
        """
        Calculate Free Spectral Range in frequency units.

        FSR_f = c / (n_g · L)

        Parameters
        ----------
        wavelength : float
            Center wavelength [m] (for reference, not used in calculation).

        Returns
        -------
        fsr_f : float
            Free Spectral Range [Hz].
        """
        return C_VACUUM / (self.geometry.n_g * self.geometry.circumference)

    def find_resonance(self, wavelength_center: float, wavelength_span: float = 50e-9,
                       n_points: int = 10000) -> float:
        """
        Find the resonance wavelength nearest to a given center wavelength.

        Parameters
        ----------
        wavelength_center : float
            Approximate center wavelength [m].
        wavelength_span : float, optional
            Search span [m]. Default: 50 nm.
        n_points : int, optional
            Number of search points.

        Returns
        -------
        resonance : float
            Resonance wavelength [m].
        """
        wavelengths = np.linspace(
            wavelength_center - wavelength_span / 2,
            wavelength_center + wavelength_span / 2,
            n_points
        )
        T = self.transmission(wavelengths)
        min_idx = np.argmin(T)
        return wavelengths[min_idx]

    def linewidth(self, wavelength: float) -> float:
        """
        Calculate the 3dB linewidth (FWHM) of the resonance.

        Analytical formula for all-pass ring:
            Δλ = (1 - ra) · λ² / (π · n_g · L · sqrt(ra))

        Parameters
        ----------
        wavelength : float
            Resonance wavelength [m].

        Returns
        -------
        linewidth : float
            3dB linewidth [m].
        """
        a = self.geometry.round_trip_loss
        r = self.geometry.self_coupling
        ra = r * a

        # FWHM formula derived from Lorentzian approximation near resonance
        numerator = (1 - ra) * wavelength**2
        denominator = np.pi * self.geometry.n_g * self.geometry.circumference * np.sqrt(ra)

        return numerator / denominator

    def quality_factor(self, wavelength: float) -> float:
        """
        Calculate the loaded quality factor Q = λ / Δλ.

        Parameters
        ----------
        wavelength : float
            Resonance wavelength [m].

        Returns
        -------
        Q : float
            Quality factor [dimensionless].
        """
        return wavelength / self.linewidth(wavelength)

    def extinction_ratio(self, wavelength: float) -> float:
        """
        Calculate the extinction ratio (ER) in dB.

        ER = T_max / T_min where T_min is at resonance.

        For an all-pass ring:
            T_min = (a - r)² / (1 - ra)²
            T_max = (a + r)² / (1 + ra)²  (at anti-resonance)

        Parameters
        ----------
        wavelength : float
            Resonance wavelength [m] (for reference).

        Returns
        -------
        ER : float
            Extinction ratio [dB].
        """
        a = self.geometry.round_trip_loss
        r = self.geometry.self_coupling

        # Transmission at resonance (φ = 2πm, cos(φ) = 1)
        T_on = (a - r)**2 / (1 - r * a)**2

        # Transmission at anti-resonance (φ = (2m+1)π, cos(φ) = -1)
        T_off = (a + r)**2 / (1 + r * a)**2

        # ER in dB (T_off / T_on, since T_on < T_off)
        return 10 * np.log10(T_off / T_on)

    def finesse(self, wavelength: float) -> float:
        """
        Calculate the finesse F = FSR / Δλ.

        Parameters
        ----------
        wavelength : float
            Resonance wavelength [m].

        Returns
        -------
        F : float
            Finesse [dimensionless].
        """
        return self.fsr(wavelength) / self.linewidth(wavelength)

    def metrics(self, wavelength: float = 1.55e-6) -> RingMetrics:
        """
        Calculate all ring metrics at a given wavelength.

        Parameters
        ----------
        wavelength : float, optional
            Center wavelength [m]. Default: 1.55 μm (C-band).

        Returns
        -------
        metrics : RingMetrics
            Dataclass containing FSR, Q, ER, finesse, etc.
        """
        fsr_m = self.fsr(wavelength)
        fsr_hz = self.fsr_frequency(wavelength)
        delta_lambda = self.linewidth(wavelength)

        return RingMetrics(
            fsr=fsr_m,
            fsr_ghz=fsr_hz / 1e9,
            linewidth=delta_lambda,
            quality_factor=wavelength / delta_lambda,
            extinction_ratio_db=self.extinction_ratio(wavelength),
            finesse=fsr_m / delta_lambda,
            resonance_wavelength=wavelength,
        )

    def spectrum(self, wavelength_center: float = 1.55e-6,
                 wavelength_span: Optional[float] = None,
                 n_points: int = 1000) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Generate transmission spectrum over a wavelength range.

        Parameters
        ----------
        wavelength_center : float, optional
            Center wavelength [m]. Default: 1.55 μm.
        wavelength_span : float, optional
            Wavelength span [m]. Default: 2×FSR.
        n_points : int, optional
            Number of wavelength points.

        Returns
        -------
        wavelengths : ndarray
            Wavelength array [m].
        transmission : ndarray
            Power transmission array.
        """
        if wavelength_span is None:
            wavelength_span = 2 * self.fsr(wavelength_center)

        wavelengths = np.linspace(
            wavelength_center - wavelength_span / 2,
            wavelength_center + wavelength_span / 2,
            n_points
        )
        T = self.transmission(wavelengths)

        return wavelengths, T

    def __repr__(self) -> str:
        g = self.geometry
        return (
            f"RingResonator(R={g.radius*1e6:.1f}μm, κ={g.kappa:.2f}, "
            f"α={g.alpha:.1f}dB/cm, n_eff={g.n_eff:.2f}, n_g={g.n_g:.2f})"
        )
