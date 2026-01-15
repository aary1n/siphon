"""
Thermal Model for Silicon Ring Resonators

Implements the thermo-optic effect and heater power budget calculations
for silicon photonic ring resonators.

Theory:
    Silicon has a large thermo-optic coefficient:
        dn/dT ≈ 1.8 × 10⁻⁴ K⁻¹

    This causes resonance wavelength shift:
        Δλ_res / ΔT = (λ / n_g) × (dn_eff/dT)

    The heater power budget defines how much electrical power is needed
    to tune the resonator to compensate for fabrication variations.

References:
    [1] Cocorullo, G. et al., "Thermo-optic coefficient of silicon," J. Appl. Phys. 74, 3271 (1993)
    [2] Xu, Q. et al., "Micrometre-scale silicon electro-optic modulator," Nature 435, 325-327 (2005)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from siphon.ring import RingResonator, RingGeometry


# Physical constants for silicon
DN_DT_SILICON = 1.8e-4  # Thermo-optic coefficient [K⁻¹]
THERMAL_RESISTANCE_TYPICAL = 2000  # Typical heater thermal resistance [K/W]


@dataclass(frozen=True)
class ThermalConfig:
    """
    Thermal model configuration parameters.

    Parameters
    ----------
    dn_dt : float
        Thermo-optic coefficient of silicon [K⁻¹].
        Default: 1.8×10⁻⁴ K⁻¹.
    thermal_resistance : float
        Heater thermal resistance R_th [K/W].
        P = ΔT / R_th gives heater power.
        Typical values: 1000-5000 K/W depending on heater design.
    ambient_temperature : float
        Ambient/substrate temperature [K].
        Default: 300 K (room temperature).
    max_heater_power : float
        Maximum allowed heater power [W].
        Default: 10 mW.
    max_temperature_rise : float
        Maximum allowed temperature rise [K].
        Default: 50 K.
    """
    dn_dt: float = DN_DT_SILICON
    thermal_resistance: float = THERMAL_RESISTANCE_TYPICAL
    ambient_temperature: float = 300.0
    max_heater_power: float = 10e-3  # 10 mW
    max_temperature_rise: float = 50.0  # 50 K

    def __post_init__(self) -> None:
        if self.dn_dt <= 0:
            raise ValueError(f"dn/dT must be positive, got {self.dn_dt}")
        if self.thermal_resistance <= 0:
            raise ValueError(f"Thermal resistance must be positive, got {self.thermal_resistance}")


@dataclass(frozen=True)
class ThermalMetrics:
    """
    Thermal tuning metrics for a ring resonator.

    Parameters
    ----------
    wavelength_shift_per_kelvin : float
        Resonance wavelength shift per Kelvin [m/K].
    temperature_per_fsr : float
        Temperature change needed to shift by one FSR [K].
    power_per_fsr : float
        Heater power needed to shift by one FSR [W].
    tuning_efficiency : float
        Wavelength tuning efficiency [m/W] or [pm/mW].
    max_tuning_range : float
        Maximum tuning range within power budget [m].
    fsr : float
        Free spectral range [m].
    """
    wavelength_shift_per_kelvin: float
    temperature_per_fsr: float
    power_per_fsr: float
    tuning_efficiency: float
    max_tuning_range: float
    fsr: float


class ThermalModel:
    """
    Thermal tuning model for silicon ring resonators.

    Calculates resonance wavelength shift due to temperature change
    and the heater power required for thermal tuning.

    Parameters
    ----------
    ring : RingResonator
        Ring resonator to model.
    config : ThermalConfig, optional
        Thermal model configuration. Uses defaults if not provided.

    Examples
    --------
    >>> geom = RingGeometry(
    ...     radius=10e-6, kappa=0.2, alpha=2.0, n_eff=2.4, n_g=4.2
    ... )
    >>> ring = RingResonator(geom)
    >>> thermal = ThermalModel(ring)
    >>> metrics = thermal.metrics(wavelength=1.55e-6)
    >>> print(f"Power to shift 1 FSR: {metrics.power_per_fsr * 1e3:.2f} mW")
    """

    def __init__(self, ring: RingResonator, config: Optional[ThermalConfig] = None) -> None:
        self.ring = ring
        self.config = config if config is not None else ThermalConfig()

    def dn_eff_dt(self) -> float:
        """
        Effective index change with temperature.

        For a simple approximation, we assume:
            dn_eff/dT ≈ Γ × dn_Si/dT

        where Γ is the confinement factor (fraction of mode in silicon).
        For typical silicon wire waveguides, Γ ≈ 0.8-0.95.

        Here we use a simplified model assuming Γ ≈ 0.9.

        Returns
        -------
        dn_eff_dt : float
            Effective index thermal coefficient [K⁻¹].
        """
        confinement_factor = 0.9  # Typical for Si wire
        return confinement_factor * self.config.dn_dt

    def wavelength_shift_per_kelvin(self, wavelength: float) -> float:
        """
        Calculate resonance wavelength shift per Kelvin.

        Δλ/ΔT = (λ / n_g) × (dn_eff/dT)

        Parameters
        ----------
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        dlambda_dT : float
            Wavelength shift per Kelvin [m/K].
        """
        return (wavelength / self.ring.geometry.n_g) * self.dn_eff_dt()

    def temperature_for_wavelength_shift(self, delta_lambda: float, wavelength: float) -> float:
        """
        Calculate temperature change needed for a given wavelength shift.

        ΔT = Δλ / (dλ/dT)

        Parameters
        ----------
        delta_lambda : float
            Desired wavelength shift [m].
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        delta_T : float
            Required temperature change [K].
        """
        dlambda_dT = self.wavelength_shift_per_kelvin(wavelength)
        return delta_lambda / dlambda_dT

    def power_for_wavelength_shift(self, delta_lambda: float, wavelength: float) -> float:
        """
        Calculate heater power needed for a given wavelength shift.

        P = ΔT / R_th

        Parameters
        ----------
        delta_lambda : float
            Desired wavelength shift [m].
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        power : float
            Required heater power [W].
        """
        delta_T = self.temperature_for_wavelength_shift(delta_lambda, wavelength)
        return delta_T / self.config.thermal_resistance

    def wavelength_shift_for_power(self, power: float, wavelength: float) -> float:
        """
        Calculate wavelength shift achieved by a given heater power.

        Δλ = P × R_th × (dλ/dT)

        Parameters
        ----------
        power : float
            Heater power [W].
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        delta_lambda : float
            Wavelength shift [m].
        """
        delta_T = power * self.config.thermal_resistance
        dlambda_dT = self.wavelength_shift_per_kelvin(wavelength)
        return delta_T * dlambda_dT

    def temperature_per_fsr(self, wavelength: float) -> float:
        """
        Calculate temperature change needed to shift by one FSR.

        This is a key metric for determining the thermal tuning budget.

        Parameters
        ----------
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        delta_T : float
            Temperature change for 1 FSR shift [K].
        """
        fsr = self.ring.fsr(wavelength)
        return self.temperature_for_wavelength_shift(fsr, wavelength)

    def power_per_fsr(self, wavelength: float) -> float:
        """
        Calculate heater power needed to shift by one FSR.

        This is the key "heater power budget" metric from the roadmap.

        Parameters
        ----------
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        power : float
            Heater power for 1 FSR shift [W].
        """
        fsr = self.ring.fsr(wavelength)
        return self.power_for_wavelength_shift(fsr, wavelength)

    def tuning_efficiency(self, wavelength: float) -> float:
        """
        Calculate wavelength tuning efficiency (pm/mW).

        Parameters
        ----------
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        efficiency : float
            Tuning efficiency [m/W].
        """
        # Power for 1 pm shift
        dlambda_dT = self.wavelength_shift_per_kelvin(wavelength)
        return dlambda_dT * self.config.thermal_resistance

    def max_tuning_range(self, wavelength: float) -> float:
        """
        Calculate maximum tuning range within power budget.

        Parameters
        ----------
        wavelength : float
            Operating wavelength [m].

        Returns
        -------
        tuning_range : float
            Maximum wavelength tuning range [m].
        """
        return self.wavelength_shift_for_power(self.config.max_heater_power, wavelength)

    def shifted_resonance(self, base_wavelength: float, delta_T: float) -> float:
        """
        Calculate new resonance wavelength after temperature change.

        Parameters
        ----------
        base_wavelength : float
            Initial resonance wavelength [m].
        delta_T : float
            Temperature change [K].

        Returns
        -------
        new_wavelength : float
            Shifted resonance wavelength [m].
        """
        dlambda_dT = self.wavelength_shift_per_kelvin(base_wavelength)
        return base_wavelength + delta_T * dlambda_dT

    def shifted_spectrum(
        self,
        wavelength_center: float,
        delta_T: float,
        wavelength_span: Optional[float] = None,
        n_points: int = 1000
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Generate transmission spectrum at elevated temperature.

        The temperature shift is modeled as an effective index change,
        which shifts all resonances by the same amount.

        Parameters
        ----------
        wavelength_center : float
            Center wavelength [m].
        delta_T : float
            Temperature change from ambient [K].
        wavelength_span : float, optional
            Wavelength span [m]. Default: 2×FSR.
        n_points : int, optional
            Number of wavelength points.

        Returns
        -------
        wavelengths : ndarray
            Wavelength array [m].
        transmission : ndarray
            Power transmission array at elevated temperature.
        """
        # Calculate wavelength shift due to temperature
        delta_lambda = delta_T * self.wavelength_shift_per_kelvin(wavelength_center)

        if wavelength_span is None:
            wavelength_span = 2 * self.ring.fsr(wavelength_center)

        wavelengths = np.linspace(
            wavelength_center - wavelength_span / 2,
            wavelength_center + wavelength_span / 2,
            n_points
        )

        # Shift the effective wavelength (equivalent to shifting n_eff)
        # T(λ, T+ΔT) ≈ T(λ - Δλ, T)
        shifted_wavelengths = wavelengths - delta_lambda
        T = self.ring.transmission(shifted_wavelengths)

        return wavelengths, T

    def metrics(self, wavelength: float = 1.55e-6) -> ThermalMetrics:
        """
        Calculate all thermal metrics at a given wavelength.

        Parameters
        ----------
        wavelength : float, optional
            Operating wavelength [m]. Default: 1.55 μm.

        Returns
        -------
        metrics : ThermalMetrics
            Dataclass containing thermal tuning parameters.
        """
        fsr = self.ring.fsr(wavelength)
        dlambda_dT = self.wavelength_shift_per_kelvin(wavelength)
        temp_per_fsr = self.temperature_per_fsr(wavelength)
        power_fsr = self.power_per_fsr(wavelength)

        return ThermalMetrics(
            wavelength_shift_per_kelvin=dlambda_dT,
            temperature_per_fsr=temp_per_fsr,
            power_per_fsr=power_fsr,
            tuning_efficiency=self.tuning_efficiency(wavelength),
            max_tuning_range=self.max_tuning_range(wavelength),
            fsr=fsr,
        )

    def __repr__(self) -> str:
        return (
            f"ThermalModel(ring={self.ring!r}, "
            f"dn/dT={self.config.dn_dt:.2e} K⁻¹, "
            f"R_th={self.config.thermal_resistance:.0f} K/W)"
        )


def estimate_thermal_resistance(heater_length: float, heater_width: float,
                                 heater_thickness: float = 100e-9,
                                 oxide_thickness: float = 2e-6) -> float:
    """
    Estimate heater thermal resistance from geometry.

    Simple 1D model: R_th ≈ t_ox / (k_ox × A_heater)

    Parameters
    ----------
    heater_length : float
        Heater length [m].
    heater_width : float
        Heater width [m].
    heater_thickness : float, optional
        Heater thickness [m]. Default: 100 nm.
    oxide_thickness : float, optional
        Oxide thickness between heater and waveguide [m]. Default: 2 μm.

    Returns
    -------
    R_th : float
        Estimated thermal resistance [K/W].

    Notes
    -----
    This is a rough estimate. Actual thermal resistance depends on:
    - 3D heat spreading
    - Substrate thermal conductivity
    - Air convection
    - Heater material properties

    Typical measured values are 1000-5000 K/W for integrated heaters.
    """
    # Thermal conductivity of SiO2
    k_oxide = 1.4  # W/(m·K)

    # Heater area
    area = heater_length * heater_width

    # Simple 1D thermal resistance
    return oxide_thickness / (k_oxide * area)
