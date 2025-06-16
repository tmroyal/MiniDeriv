from numba import njit
import numpy as np


# TODO: Stable dudx for linear advection; i.e. upwind


@njit
def dudx(u: np.ndarray, i: int, dx: float) -> float:
    """
    Compute the first derivative of a 1D array at index i using central differences.

    Periodic boundary conditions are applied: this must be overwritten for other boundary conditions.

    Args:
        u (np.ndarray): Input array for which the derivative is computed.
        i (int): Index at which the derivative is computed.
        dx (float): Spatial step size.

    Returns:
        float: The computed first derivative at index i.
    """
    left = (i - 1) % len(u)
    right = (i + 1) % len(u)
    return (u[right] - u[left]) / (2 * dx)


@njit
def laplacian(u: np.ndarray, i: int, dx: float) -> float:
    """
    Compute the Laplacian of a 1D array at index i using central differences.

    Periodic boundary conditions are applied: this must be overwritten for other boundary conditions.

    Args:
        u (np.ndarray): Input array for which the Laplacian is computed.
        i (int): Index at which the Laplacian is computed.
        dx (float): Spatial step size.

    Returns:
        float: The computed Laplacian at index i.
    """
    left = (i - 1) % len(u)
    right = (i + 1) % len(u)
    return (u[right] - 2 * u[i] + u[left]) / (dx * dx)


lap = laplacian


@njit
def d3udx3(u: np.ndarray, i: int, dx: float) -> float:
    """
    Compute the third derivative of a 1D array at index i using central differences.

    Periodic boundary conditions are applied: this must be overwritten for other boundary conditions.

    Args:
        u (np.ndarray): Input array for which the third derivative is computed.
        i (int): Index at which the third derivative is computed.
        dx (float): Spatial step size.

    Returns:
        float: The computed third derivative at index i.
    """
    N = len(u)
    im2 = (i - 2) % N
    im1 = (i - 1) % N
    ip1 = (i + 1) % N
    ip2 = (i + 2) % N
    return (u[ip2] - 2 * u[ip1] + 2 * u[im1] - u[im2]) / (2 * dx**3)


d3 = d3udx3


@njit
def d4udx4(u: np.ndarray, i: int, dx: float) -> float:
    """
    Compute the fourth derivative of a 1D array at index i using central differences.

    Periodic boundary conditions are applied: this must be overwritten for other boundary conditions.

    Args:
        u (np.ndarray): Input array for which the fourth derivative is computed.
        i (int): Index at which the fourth derivative is computed.
        dx (float): Spatial step size.

    Returns:
        float: The computed fourth derivative at index i.
    """
    left = (i - 1) % len(u)
    left_2 = (left - 1) % len(u)
    right = (i + 1) % len(u)
    right_2 = (right + 1) % len(u)
    return (u[right_2] - 4 * u[right] + 6 * u[i] - 4 * u[left] + u[left_2]) / (dx * dx * dx * dx)


d4 = d4udx4
