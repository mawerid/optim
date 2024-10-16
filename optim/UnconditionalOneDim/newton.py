from typing import Callable
import numpy as np
from sympy import diff


def derivative(func: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """
    Calculates the derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[float], float]): The function to differentiate.
    x (float): The point at which to calculate the derivative.
    eps (float): The precision of the calculation.

    Returns:
    float: The derivative of the function at the given point.
    """
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def second_derivative(func: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """
    Calculates the second order derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[float], float]): The function to differentiate.
    x (float): The point at which to calculate the derivative.
    eps (float): The precision of the calculation.

    Returns:
    float: The second order derivative of the function at the given point.
    """
    return (func(x + eps) - 2 * func(x) + func(x - eps)) / (eps ** 2)


def find_min(func: Callable[[float], float], init_point: float,
             eps: float = 1e-3, max_iter: int = 1e2, verbose: bool = False,
             return_argmin: bool = False) -> float:
    """
    Finds the minimum value of a function within a given interval using the Newton method.

    Parameters:
    func (Callable[[float], float]): The function to minimize.
    init_point (float): The initial point to start optimization.
    eps (float): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    float: The minimum value of the function or the x value that minimizes the function.
    """
    iteration = 0

    x = init_point
    x_derivative = derivative(func, x, eps if eps < 1e-5 else 1e-5)

    while np.abs(x_derivative) > eps and iteration < max_iter:
        x_second_derivative = second_derivative(func, x, eps if eps < 1e-5 else 1e-5)
        if x_second_derivative == 0:
            break
        # Sorry, without this check it finds only nearest extremum, not minimum
        # TODO: fix Newton finding only nearest extremum
        if func(x) < func(x - x_derivative / x_second_derivative):
            x += x_derivative / x_second_derivative
        else:
            x -= x_derivative / x_second_derivative
        if verbose:
            print(
                f"Round: {iteration},\tx: {x},\tF'(x): {x_derivative},\tF''(x): {x_second_derivative}")
        iteration += 1
        x_derivative = derivative(func, x, eps if eps < 1e-5 else 1e-5)

    return x if return_argmin else func(x)


def find_max(func: Callable[[float], float], init_point: float,
             eps: float = 1e-3, max_iter: int = 1e2, verbose: bool = False,
             return_argmin: bool = False) -> float:
    """
    Finds the maximum value of a function within a given interval using the Newton method.

    Parameters:
    func (Callable[[float], float]): The function to maximize.
    interval (np.ndarray): The interval [start, end] within which to search for the maximum.
    eps (float): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

    Returns:
    float: The maximum value of the function or the x value that maximizes the function.
    """
    rev_func = lambda x: (-1) * func(x)
    return find_min(rev_func, init_point, eps, max_iter, verbose, return_argmin)
