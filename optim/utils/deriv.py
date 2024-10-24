from typing import Callable

import numpy as np


def derivative(func: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """Calculates the derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[float], float]): The function to differentiate.
    x (float): The point at which to calculate the derivative.
    eps (float): The precision of the calculation.

    Returns:
    float: The derivative of the function at the given point.
    """
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def second_derivative(func: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """Calculates the second order derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[float], float]): The function to differentiate.
    x (float): The point at which to calculate the derivative.
    eps (float): The precision of the calculation.

    Returns:
    float: The second order derivative of the function at the given point.
    """
    return (func(x + eps) - 2 * func(x) + func(x - eps)) / (eps ** 2)


def gradient(func: Callable[[np.ndarray], float], x: np.ndarray, ndim: int = 1, eps: float = 1e-5) -> np.ndarray:
    """Calculates the derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[np.ndarray], float]): The function to differentiate.
    x (np.ndarray): The point at which to calculate the derivative.
    ndim (int): The number of dimensions of the function.
    eps (float): The precision of the calculation.

    Returns:
    np.ndarray: The derivative of the function at the given point.
    """
    x_plus = np.tile(x, (ndim, 1)) + np.eye(ndim) * eps
    x_minus = np.tile(x, (ndim, 1)) - np.eye(ndim) * eps
    grad = (np.apply_along_axis(func, 1, x_plus) - np.apply_along_axis(func, 1, x_minus)) / (2 * eps)
    return grad


def jacobian(func: Callable[[np.ndarray], float], x: np.ndarray, ndim: int = 1, eps: float = 1e-5) -> np.ndarray:
    """Calculates the Jacobian matrix of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[np.ndarray], float]): The function to differentiate.
    x (np.ndarray): The point at which to calculate the derivative.
    ndim (int): The number of dimensions of the function.
    eps (float): The precision of the calculation.

    Returns:
    np.ndarray: The derivative of the function at the given point.
    """
    x_plus = np.tile(x, (ndim, 1)) + np.eye(ndim) * eps
    x_minus = np.tile(x, (ndim, 1)) - np.eye(ndim) * eps
    # TODO Return Jacobian
    grad = (np.apply_along_axis(func, 1, x_plus) - np.apply_along_axis(func, 1, x_minus)) / (2 * eps)
    return grad


def hessian(func: Callable[[np.ndarray], float], x: np.ndarray, ndim: int = 1, eps: float = 1e-5) -> np.ndarray:
    """Calculates the Hessian matrix of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[np.ndarray], float]): The function to differentiate.
    x (np.ndarray): The point at which to calculate the derivative.
    ndim (int): The number of dimensions of the function.
    eps (float): The precision of the calculation.

    Returns:
    np.ndarray: The derivative of the function at the given point.
    """
    x_plus = np.tile(x, (ndim, 1)) + np.eye(ndim) * eps
    x_minus = np.tile(x, (ndim, 1)) - np.eye(ndim) * eps
    # TODO Return Hessian
    grad = (np.apply_along_axis(func, 1, x_plus) - np.apply_along_axis(func, 1, x_minus)) / (2 * eps)
    return grad
