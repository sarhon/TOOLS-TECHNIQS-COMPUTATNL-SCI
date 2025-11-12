"""
PyDiscord - Python Discrete Ordinates Transport Solver

A discrete ordinates (SN) solver for 1D neutron transport problems.
"""

from .solver import (
    Material,
    Settings,
    solve_flux,
    avg_flux,
    RightFlux,
    LeftFlux,
    Matrix,
)

__version__ = "0.1.0"
__all__ = [
    "Material",
    "Settings",
    "solve_flux",
    "avg_flux",
    "RightFlux",
    "LeftFlux",
    "Matrix",
]
