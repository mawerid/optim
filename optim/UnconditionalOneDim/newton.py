from typing import Callable
import numpy as np


def derivative(func: Callable[[np.float64], np.float64], x: np.float64, eps: np.float64 = 1e-3) -> np.float64:
    """
    Calculates the derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to differentiate.
    x (np.float64): The point at which to calculate the derivative.
    eps (np.float64): The precision of the calculation.

    Returns:
    np.float64: The derivative of the function at the given point.
    """
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def second_derivative(func: Callable[[np.float64], np.float64], x: np.float64, eps: np.float64 = 1e-3) -> np.float64:
    """
    Calculates the second order derivative of a function at a given point using the central difference method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to differentiate.
    x (np.float64): The point at which to calculate the derivative.
    eps (np.float64): The precision of the calculation.

    Returns:
    np.float64: The second order derivative of the function at the given point.
    """
    return (func(x + eps) - 2 * func(x) + func(x - eps)) / (eps * eps)


def find_min(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the minimum value of a function within a given interval using the Newton method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to minimize.
    interval (np.ndarray): The interval [start, end] within which to search for the minimum.
    eps (np.float64): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    np.float64: The minimum value of the function or the x value that minimizes the function.
    """
    start: np.float64 = interval[0]
    end: np.float64 = interval[1]

    iteration = 1
    x = (end - start) / 2
    x_derivative = derivative(func, x, eps)

    while np.abs(x_derivative) > eps:
        start_derivarive = derivative(func, start, eps)
        start_second_derivative = second_derivative(func, start, eps)
        end_derivarive = derivative(func, end, eps)
        if start_derivarive * end_derivarive > 0:
            raise ValueError("The function has no minimum within the given interval.")
        x = (start - start_derivarive
             / start_second_derivative)
        x_derivative = derivative(func, x, eps)
        if start_derivarive * x_derivative < 0:
            end = x
        elif x_derivative * end_derivarive < 0:
            start = x
        if verbose:
            print(
                f"Round: {iteration},\tInterval: [{start}, {end}],\tx: {x},\tF'(x): {x_derivative}")
        iteration += 1

    if return_argmin:
        return x
    else:
        return func(x)


def find_max(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the maximum value of a function within a given interval using the Newton method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to maximize.
    interval (np.ndarray): The interval [start, end] within which to search for the maximum.
    eps (np.float64): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

    Returns:
    np.float64: The maximum value of the function or the x value that maximizes the function.
    """
    rev_func = lambda x: (-1) * func(x)
    return find_min(rev_func, interval, eps, verbose, return_argmin)
