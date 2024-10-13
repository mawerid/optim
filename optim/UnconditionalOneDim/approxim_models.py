from typing import Callable
import numpy as np


def find_min(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the minimum value of a function within a given interval using the approximation model method.

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
    while np.abs(end - start) > eps:
        middle = (start + end) / 2

        a_matrix = np.array([[np.power(start, p) for p in range(3)],
                             [np.power(middle, p) for p in range(3)],
                             [np.power(end, p) for p in range(3)]], dtype=np.float64)
        b_vector = np.array([func(start), func(middle), func(end)], dtype=np.float64)
        coefficients = np.linalg.solve(a_matrix, b_vector)
        extremum = -coefficients[1] / (2 * coefficients[2])
        if np.isnan(extremum):
            print("Warning: extremum is NaN")
            break

        start, end = (extremum, middle) if extremum < middle else (middle, extremum)

        if verbose:
            print(f"Iteration: {iteration}, interval: [{start}, {end}], extremum: {extremum}")
        iteration += 1

    if return_argmin:
        return (start + end) / 2
    else:
        return func((start + end) / 2)


def find_max(func: Callable[[np.float64], np.float64], interval: np.ndarray,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the maximum value of a function within a given interval using the approximation model method.

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
