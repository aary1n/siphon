"""
SiPhON - Silicon Photonics with Open Numerics

A high-performance simulation framework for silicon photonic ring resonators,
demonstrating the "Physics -> Numerics -> Yield" workflow.

Version: 0.2-dev (Variability & Yield Architecture)
"""

__version__ = "0.2.0-dev"
__author__ = "Aaryan Sharif"

from siphon.ring import RingResonator
from siphon.thermal import ThermalModel
from siphon.sensitivity import EffectiveIndexSolver
from siphon.variability import YieldAnalyzer

__all__ = ["RingResonator", "ThermalModel", "EffectiveIndexSolver", "YieldAnalyzer"]
