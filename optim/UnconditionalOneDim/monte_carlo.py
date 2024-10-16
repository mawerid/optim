from typing import Callable
import numpy as np


def find_min(func: Callable[[float], float], interval: np.ndarray, point_num: np.int64,
             verbose: bool = False, return_argmin: bool = False) -> float:
    """
    Finds the minimum value of a function within a given interval using the Monte-Carlo method.

    Parameters:
    func (Callable[[float], float]): The function to minimize.
    interval (np.ndarray): The interval [start, end] within which to search for the minimum.
    point_num (np.int64): The number of points to generate within the interval.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    float: The minimum value of the function or the x value that minimizes the function.
    """
    start: float = interval[0]
    end: float = interval[1]
    points = np.linspace(start, end, num=point_num, dtype=float)
    min_k = np.argmin(func(points))
    if verbose:
        print(f"K of minimal F value: {points[min_k]},\tF(K): {func(points[min_k])}")

    return points[min_k] if return_argmin else func(points[min_k])


def find_max(func: Callable[[float], float], interval: np.ndarray, point_num: np.int64,
             verbose: bool = False, return_argmin: bool = False) -> float:
    """
    Finds the maximum value of a function within a given interval using the Monte-Carlo method.

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
    return find_min(rev_func, interval, point_num, verbose, return_argmin)
