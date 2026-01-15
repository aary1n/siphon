"""
SiPhON - Silicon Photonics with Open Numerics

A high-performance simulation framework for silicon photonic ring resonators,
demonstrating the "Physics → Numerics → Yield" workflow.

Version: 0.1-dev (Core Physics Baseline)
"""

__version__ = "0.1.0-dev"
__author__ = "Aaryan Sharif"

from siphon.ring import RingResonator
from siphon.thermal import ThermalModel

__all__ = ["RingResonator", "ThermalModel"]
