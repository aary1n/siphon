"""
Monte Carlo Variability Engine and Yield Analysis

Connects fabrication process variation to thermal power budget via:
1. Sampling geometry variations (w, h) from multivariate normal
2. Mapping to effective index deviations via sensitivity coefficients
3. Mapping to resonance wavelength shifts via chain rule
4. Computing required heater power via thermal model
5. Calculating yield as fraction of devices within power budget

Theory:
    For small geometry variations (delta_w, delta_h), the linear approximation:
        delta_n_eff = (dn_eff/dw) * delta_w + (dn_eff/dh) * delta_h

    Resonance wavelength shift (chain rule):
        delta_lambda = (lambda / n_g) * delta_n_eff

    Required heater power (worst-case):
        P_heater = |delta_lambda| / tuning_efficiency

    Yield metric (Thermal Overhead):
        Yield = fraction of devices with P_heater < P_max

    This is a conservative estimate because:
    - Heaters can only red-shift (increase temperature)
    - A device already red-shifted may need less power
    - FSR-modular tuning is not considered (Phase 0.4 refinement)

References:
    [1] Lu, Z. et al., "Performance prediction for silicon photonics integrated circuits
        with layout-dependent correlated manufacturing variability," Opt. Express 25, 9712 (2017)
    [2] Selvaraja, S.K. et al., "Subnanometer linewidth uniformity in silicon nanophotonic
        waveguide devices using CMOS fabrication technology," IEEE JSTQE 16, 316-324 (2010)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from siphon.ring import RingResonator, RingGeometry
from siphon.thermal import ThermalModel, ThermalConfig
from siphon.sensitivity import (
    SensitivityCoefficients,
    EffectiveIndexSolver,
    WaveguideGeometry,
)


@dataclass(frozen=True)
class FabricationConfig:
    """
    Fabrication process variation parameters.

    Defines the statistical distributions for waveguide geometry variations
    due to lithography, etching, and deposition non-uniformity.

    Parameters
    ----------
    w_nominal : float
        Nominal waveguide width [m]. Default: 500e-9 (500nm).
    h_nominal : float
        Nominal silicon thickness [m]. Default: 220e-9 (220nm).
    sigma_w : float
        Width variation 1-sigma standard deviation [m]. Default: 10e-9 (10nm).
    sigma_h : float
        Height variation 1-sigma standard deviation [m]. Default: 5e-9 (5nm).
    correlation : float
        Correlation between width and height variations [-1, 1].
        Default: 0.0 (independent variations).

    Notes
    -----
    Typical foundry tolerances (1-sigma):
    - Width: 5-15 nm (lithography + etch)
    - Height: 2-10 nm (SOI wafer + thinning)
    Sources: IMEC MPW specs, Selvaraja et al. IEEE JSTQE 2010.
    """
    w_nominal: float = 500e-9
    h_nominal: float = 220e-9
    sigma_w: float = 10e-9
    sigma_h: float = 5e-9
    correlation: float = 0.0

    def __post_init__(self) -> None:
        """Validate fabrication parameters."""
        if self.w_nominal <= 0:
            raise ValueError(f"Nominal width must be positive, got {self.w_nominal}")
        if self.h_nominal <= 0:
            raise ValueError(f"Nominal height must be positive, got {self.h_nominal}")
        if self.sigma_w < 0:
            raise ValueError(f"Width sigma must be non-negative, got {self.sigma_w}")
        if self.sigma_h < 0:
            raise ValueError(f"Height sigma must be non-negative, got {self.sigma_h}")
        if not -1 <= self.correlation <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {self.correlation}")

    @property
    def covariance_matrix(self) -> NDArray[np.floating]:
        """
        2x2 covariance matrix for the (w, h) joint distribution.

        Returns
        -------
        cov : ndarray, shape (2, 2)
            [[sigma_w^2, rho*sigma_w*sigma_h],
             [rho*sigma_w*sigma_h, sigma_h^2]]
        """
        cov_wh = self.correlation * self.sigma_w * self.sigma_h
        return np.array([
            [self.sigma_w**2, cov_wh],
            [cov_wh, self.sigma_h**2],
        ])


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Monte Carlo sampling configuration.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples. Default: 10_000.
    seed : int or None
        Random seed for reproducibility. Default: 42.
        Set to None for non-reproducible runs.
    max_heater_power : float
        Maximum allowed heater power [W]. Default: 10e-3 (10mW).
    target_wavelength : float
        Target resonance wavelength [m]. Default: 1.55e-6 (C-band).
    """
    n_samples: int = 10_000
    seed: Optional[int] = 42
    max_heater_power: float = 10e-3
    target_wavelength: float = 1.55e-6

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.max_heater_power <= 0:
            raise ValueError(f"max_heater_power must be positive, got {self.max_heater_power}")
        if self.target_wavelength <= 0:
            raise ValueError(
                f"target_wavelength must be positive, got {self.target_wavelength}"
            )


@dataclass(frozen=True)
class YieldResult:
    """
    Monte Carlo yield analysis results.

    Contains both the scalar yield metric and full distribution arrays
    for detailed post-analysis and visualization.

    Parameters
    ----------
    yield_fraction : float
        Fraction of devices within power budget [0, 1].
    yield_percent : float
        Yield as percentage [0, 100].
    heater_powers : NDArray
        Required heater power for each sample [W], shape (n_samples,).
    wavelength_shifts : NDArray
        Resonance wavelength shift for each sample [m], shape (n_samples,).
    delta_n_eff : NDArray
        Effective index deviation for each sample, shape (n_samples,).
    width_samples : NDArray
        Sampled waveguide widths [m], shape (n_samples,).
    height_samples : NDArray
        Sampled silicon heights [m], shape (n_samples,).
    n_samples : int
        Number of MC samples used.
    max_heater_power : float
        Power budget threshold [W].
    mean_heater_power : float
        Mean required heater power [W].
    std_heater_power : float
        Standard deviation of heater power [W].
    p95_heater_power : float
        95th percentile heater power [W].
    """
    yield_fraction: float
    yield_percent: float
    heater_powers: NDArray[np.floating]
    wavelength_shifts: NDArray[np.floating]
    delta_n_eff: NDArray[np.floating]
    width_samples: NDArray[np.floating]
    height_samples: NDArray[np.floating]
    n_samples: int
    max_heater_power: float
    mean_heater_power: float
    std_heater_power: float
    p95_heater_power: float


class YieldAnalyzer:
    """
    Monte Carlo yield analysis for ring resonator fabrication variability.

    Connects the full pipeline:
        Process variation -> n_eff deviation -> wavelength shift -> heater power -> yield

    All array operations are vectorized with NumPy (no Python for-loops in the
    hot path). 10,000 samples run in well under 1 second.

    Parameters
    ----------
    ring : RingResonator
        Ring resonator model (provides n_g for chain rule).
    thermal : ThermalModel
        Thermal model (provides tuning efficiency for power conversion).
    sensitivity : SensitivityCoefficients
        Sensitivity coefficients from EffectiveIndexSolver.
    fab_config : FabricationConfig, optional
        Fabrication tolerance parameters. Uses defaults if not provided.
    mc_config : MonteCarloConfig, optional
        Monte Carlo sampling configuration. Uses defaults if not provided.

    Examples
    --------
    >>> from siphon.ring import RingResonator, RingGeometry
    >>> from siphon.thermal import ThermalModel
    >>> from siphon.sensitivity import EffectiveIndexSolver, WaveguideGeometry
    >>> wg = WaveguideGeometry(width=500e-9, height=220e-9)
    >>> sens = EffectiveIndexSolver(wg).sensitivity(n_g=4.2)
    >>> geom = RingGeometry(radius=10e-6, kappa=0.2, alpha=2.0,
    ...                     n_eff=sens.n_eff, n_g=4.2)
    >>> ring = RingResonator(geom)
    >>> thermal = ThermalModel(ring)
    >>> analyzer = YieldAnalyzer(ring, thermal, sens)
    >>> result = analyzer.run()
    >>> print(f"Yield: {result.yield_percent:.1f}%")
    """

    def __init__(
        self,
        ring: RingResonator,
        thermal: ThermalModel,
        sensitivity: SensitivityCoefficients,
        fab_config: Optional[FabricationConfig] = None,
        mc_config: Optional[MonteCarloConfig] = None,
    ) -> None:
        self.ring = ring
        self.thermal = thermal
        self.sensitivity = sensitivity
        self.fab_config = fab_config if fab_config is not None else FabricationConfig()
        self.mc_config = mc_config if mc_config is not None else MonteCarloConfig()

    def _sample_geometry(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Sample (width, height) pairs from multivariate normal.

        Returns
        -------
        widths : ndarray, shape (n_samples,)
            Sampled waveguide widths [m].
        heights : ndarray, shape (n_samples,)
            Sampled silicon heights [m].
        """
        rng = np.random.default_rng(self.mc_config.seed)
        mean = np.array([self.fab_config.w_nominal, self.fab_config.h_nominal])
        cov = self.fab_config.covariance_matrix

        samples = rng.multivariate_normal(mean, cov, size=self.mc_config.n_samples)
        return samples[:, 0], samples[:, 1]

    def _compute_delta_n_eff(
        self,
        delta_w: NDArray[np.floating],
        delta_h: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Linear approximation for effective index deviation.

        delta_n_eff = (dn_eff/dw) * delta_w + (dn_eff/dh) * delta_h

        Parameters
        ----------
        delta_w : ndarray
            Width deviations from nominal [m].
        delta_h : ndarray
            Height deviations from nominal [m].

        Returns
        -------
        delta_n_eff : ndarray
            Effective index deviations.
        """
        return (
            self.sensitivity.dn_eff_dw * delta_w
            + self.sensitivity.dn_eff_dh * delta_h
        )

    def _compute_wavelength_shifts(
        self,
        delta_n_eff: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Map delta_n_eff to resonance wavelength shift.

        delta_lambda = (lambda / n_g) * delta_n_eff

        Parameters
        ----------
        delta_n_eff : ndarray
            Effective index deviations.

        Returns
        -------
        wavelength_shifts : ndarray
            Resonance wavelength shifts [m].
        """
        wl = self.mc_config.target_wavelength
        n_g = self.ring.geometry.n_g
        return (wl / n_g) * delta_n_eff

    def _compute_heater_powers(
        self,
        wavelength_shifts: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Map wavelength shifts to required heater power.

        Uses the conservative worst-case model:
            P_heater = |delta_lambda| / tuning_efficiency

        This overestimates power because it treats any deviation (positive
        or negative) as requiring active correction. A more refined model
        (Phase 0.4) would consider FSR-modular tuning and unidirectional
        heating constraints.

        Parameters
        ----------
        wavelength_shifts : ndarray
            Resonance wavelength shifts [m].

        Returns
        -------
        heater_powers : ndarray
            Required heater power [W].
        """
        wl = self.mc_config.target_wavelength
        efficiency = self.thermal.tuning_efficiency(wl)
        return np.abs(wavelength_shifts) / efficiency

    def run(self) -> YieldResult:
        """
        Execute Monte Carlo yield analysis.

        Pipeline:
        1. Sample (w_i, h_i) from multivariate normal
        2. Compute delta_n_eff via linear sensitivity model
        3. Compute delta_lambda via chain rule
        4. Compute P_heater via thermal model
        5. Calculate yield = fraction(P_heater < P_max)

        Returns
        -------
        result : YieldResult
            Complete yield analysis results including distributions.
        """
        # 1. Sample geometry variations
        widths, heights = self._sample_geometry()
        delta_w = widths - self.fab_config.w_nominal
        delta_h = heights - self.fab_config.h_nominal

        # 2. Compute effective index deviations (vectorized)
        delta_n_eff = self._compute_delta_n_eff(delta_w, delta_h)

        # 3. Compute wavelength shifts (vectorized)
        wavelength_shifts = self._compute_wavelength_shifts(delta_n_eff)

        # 4. Compute heater powers (vectorized)
        heater_powers = self._compute_heater_powers(wavelength_shifts)

        # 5. Calculate yield
        within_budget = heater_powers <= self.mc_config.max_heater_power
        yield_fraction = float(np.mean(within_budget))

        return YieldResult(
            yield_fraction=yield_fraction,
            yield_percent=yield_fraction * 100,
            heater_powers=heater_powers,
            wavelength_shifts=wavelength_shifts,
            delta_n_eff=delta_n_eff,
            width_samples=widths,
            height_samples=heights,
            n_samples=self.mc_config.n_samples,
            max_heater_power=self.mc_config.max_heater_power,
            mean_heater_power=float(np.mean(heater_powers)),
            std_heater_power=float(np.std(heater_powers)),
            p95_heater_power=float(np.percentile(heater_powers, 95)),
        )

    def sweep_tolerance(
        self,
        sigma_w_range: Optional[NDArray[np.floating]] = None,
        sigma_h_range: Optional[NDArray[np.floating]] = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Sweep yield as a function of fabrication tolerance.

        Runs MC analysis at each sigma value to produce yield curves.
        If both sigma ranges are None, sweeps sigma_w with fixed sigma_h.

        Parameters
        ----------
        sigma_w_range : ndarray, optional
            Width sigma values to sweep [m].
            Default: np.linspace(1e-9, 20e-9, 20).
        sigma_h_range : ndarray, optional
            Height sigma values to sweep [m].
            Default: fixed at self.fab_config.sigma_h.

        Returns
        -------
        sigma_values : ndarray
            Swept sigma_w values [m].
        yields : ndarray
            Yield fraction [0, 1] at each sigma value.
        mean_powers : ndarray
            Mean heater power [W] at each sigma value.
        """
        if sigma_w_range is None:
            sigma_w_range = np.linspace(1e-9, 20e-9, 20)

        yields = np.zeros(len(sigma_w_range))
        mean_powers = np.zeros(len(sigma_w_range))

        for i, sigma_w in enumerate(sigma_w_range):
            sigma_h = (
                sigma_h_range[i] if sigma_h_range is not None
                else self.fab_config.sigma_h
            )

            fab = FabricationConfig(
                w_nominal=self.fab_config.w_nominal,
                h_nominal=self.fab_config.h_nominal,
                sigma_w=sigma_w,
                sigma_h=sigma_h,
                correlation=self.fab_config.correlation,
            )

            analyzer = YieldAnalyzer(
                ring=self.ring,
                thermal=self.thermal,
                sensitivity=self.sensitivity,
                fab_config=fab,
                mc_config=self.mc_config,
            )
            result = analyzer.run()
            yields[i] = result.yield_fraction
            mean_powers[i] = result.mean_heater_power

        return sigma_w_range, yields, mean_powers

    def __repr__(self) -> str:
        return (
            f"YieldAnalyzer(n_samples={self.mc_config.n_samples}, "
            f"sigma_w={self.fab_config.sigma_w*1e9:.0f}nm, "
            f"sigma_h={self.fab_config.sigma_h*1e9:.0f}nm, "
            f"P_max={self.mc_config.max_heater_power*1e3:.0f}mW)"
        )


def quick_yield(
    radius: float = 10e-6,
    kappa: float = 0.2,
    alpha: float = 2.0,
    n_g: float = 4.2,
    w_nominal: float = 500e-9,
    h_nominal: float = 220e-9,
    sigma_w: float = 10e-9,
    sigma_h: float = 5e-9,
    n_samples: int = 10_000,
    max_power: float = 10e-3,
    seed: int = 42,
) -> YieldResult:
    """
    One-call convenience function for quick yield estimates.

    Creates all intermediate objects (waveguide solver, ring, thermal,
    yield analyzer) from scalar parameters.

    Parameters
    ----------
    radius : float
        Ring radius [m]. Default: 10 um.
    kappa : float
        Coupling coefficient. Default: 0.2.
    alpha : float
        Propagation loss [dB/cm]. Default: 2.0.
    n_g : float
        Group index. Default: 4.2.
    w_nominal : float
        Nominal waveguide width [m]. Default: 500nm.
    h_nominal : float
        Nominal silicon height [m]. Default: 220nm.
    sigma_w : float
        Width 1-sigma variation [m]. Default: 10nm.
    sigma_h : float
        Height 1-sigma variation [m]. Default: 5nm.
    n_samples : int
        MC sample count. Default: 10,000.
    max_power : float
        Max heater power [W]. Default: 10mW.
    seed : int
        Random seed. Default: 42.

    Returns
    -------
    result : YieldResult
        Complete yield analysis results.
    """
    # 1. Compute n_eff and sensitivities
    wg = WaveguideGeometry(width=w_nominal, height=h_nominal)
    solver = EffectiveIndexSolver(wg)
    sens = solver.sensitivity(n_g=n_g)

    # 2. Build ring and thermal models using EIM n_eff
    geom = RingGeometry(
        radius=radius,
        kappa=kappa,
        alpha=alpha,
        n_eff=sens.n_eff,
        n_g=n_g,
    )
    ring = RingResonator(geom)
    thermal = ThermalModel(ring)

    # 3. Configure and run
    fab = FabricationConfig(
        w_nominal=w_nominal,
        h_nominal=h_nominal,
        sigma_w=sigma_w,
        sigma_h=sigma_h,
    )
    mc = MonteCarloConfig(
        n_samples=n_samples,
        seed=seed,
        max_heater_power=max_power,
    )

    analyzer = YieldAnalyzer(ring, thermal, sens, fab, mc)
    return analyzer.run()
