"""Inexact projected gradient methods for Gromov-Wasserstein transport."""

from .projection import proj_transport_polytope
from .solver import ProjectionConvergenceWarning, gwipg

__all__ = ["gwipg", "proj_transport_polytope", "ProjectionConvergenceWarning"]
__version__ = "0.1.0"
