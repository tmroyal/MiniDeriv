from typing import Callable
import numpy as np

DEFAULT_DT = 1 / 22050
FirstOrderPDE = Callable[[np.ndarray, int, float], float]


class SolverError(Exception):
    """Base class for exceptions in this module."""

    pass


class SolverInitializationError(SolverError):
    """Exception raised for errors in the initialization of the solver."""

    pass


class SolverExecutionError(SolverError):
    """Exception raised for errors during the execution of the solver."""

    pass


class Storage: ...


class SolverBase:
    pass


class PDESolver(SolverBase):
    """
    Solves a first-order PDE provided by the user.
    """

    def __init__(
        self,
        domain: np.ndarray,
        rhs: FirstOrderPDE,
        dt: float = DEFAULT_DT,
        dx: str | float = "auto",
    ):
        self.domain = domain
        self.rhs = rhs
        self.dt = dt

        self.dx = GridSpacer(self).get_dx() if dx == "auto" else dx

    def solve(self, storage: Storage) -> None:
        """
        Solve the PDE and return the solution.

        Args:
            storage: Storage object to store the results.
        """
        if not storage or not isinstance(storage, Storage):
            raise SolverExecutionError("Storage must be an instance of Storage class.")
        pass


class PDEMultiSolver(SolverBase):
    """
    Solves multiple first-order PDEs provided by the user.
    """

    pass


class GridSpacer:
    """
    A class to determine the grid spacing for a PDE Solver.
    """

    def __init__(self, solver: SolverBase):
        self.solve = solver

    def get_dx(self) -> float:
        """
        Get the grid spacing for the PDE solver.

        Returns:
            float: The grid spacing.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
