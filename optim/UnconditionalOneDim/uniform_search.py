from typing import Callable
import numpy as np


def find_min(func: Callable[[np.float64], np.float64], interval: np.ndarray, inter_count: np.int64,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the minimum value of a function within a given interval using the uniform search method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to minimize.
    interval (np.ndarray): The interval [start, end] within which to search for the minimum.
    inter_count (np.int64) : The count of subintervals to divide main interval each iteration
    eps (np.float64): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    np.float64: The minimum value of the function or the x value that minimizes the function.
    """
    start: np.float64 = interval[0]
    end: np.float64 = interval[1]
    iteration = 1
    vec_func = np.vectorize(func)

    while np.abs(start - end) >= eps:
        nodes = np.linspace(start, end, num=inter_count, dtype=np.float64)
        min_k = np.argmin(vec_func(nodes))
        if min_k == inter_count:
            min_k -= 1
        elif min_k == 0:
            min_k += 1
        start, end = nodes[min_k - 1], nodes[min_k + 1]

        if verbose:
            print(
                f"Round: {iteration},\tInterval: [{start}, {end}],\tK of minimal F value: {min_k},\tF(K): {func(min_k)}")
        iteration += 1
    if return_argmin:
        return (start + end) / 2
    else:
        return func((start + end) / 2)


def find_max(func: Callable[[np.float64], np.float64], interval: np.ndarray, inter_count: np.int64,
             eps: np.float64 = 1e-3, verbose: bool = False, return_argmin: bool = False) -> np.float64:
    """
    Finds the maximum value of a function within a given interval using the uniform search method.

    Parameters:
    func (Callable[[np.float64], np.float64]): The function to maximize.
    interval (np.ndarray): The interval [start, end] within which to search for the maximum.
    inter_count (np.int64) : The count of subintervals to divide main interval each iteration
    eps (np.float64): The precision of the search.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

    Returns:
    np.float64: The maximum value of the function or the x value that maximizes the function.
    """
    rev_func = lambda x: (-1) * func(x)
    return find_min(rev_func, interval, inter_count, eps, verbose, return_argmin)
