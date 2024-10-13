from typing import Callable
import numpy as np


def find_min(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the minimum value of a function within a given interval using the dichotomy method.

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
    while np.abs(start - end) >= eps:
        x_0 = (start + end) / 2
        x_1 = x_0 - eps / 2
        x_2 = x_0 + eps / 2
        func_1 = func(x_1)
        func_2 = func(x_2)
        start, end = (start, x_0) if func_1 < func_2 else (x_0, end)

        if verbose:
            print(f"Round: {iteration},\tx_0: {x_0},\tx_1: {x_1},\tx_2: {x_2},\tF(x_1): {func_1},\tF(x_2): {func_2}")
        iteration += 1
    if return_argmin:
        return (start + end) / 2
    else:
        return func((start + end) / 2)


def find_max(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the maximum value of a function within a given interval using the dichotomy method.

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