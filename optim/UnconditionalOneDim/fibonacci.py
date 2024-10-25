"""Fibonacci method for unconstrained optimization.
"""


from typing import Callable

import numpy as np


def fibonacci(n: int) -> int:
    """Returns the n-th Fibonacci number.

    Parameters:
        n (int): The index of the Fibonacci number to return.

    Returns:
        int: The n-th Fibonacci number.
    """
    golden_ratio = float((1 + np.sqrt(5)) / 2)
    return int((np.power(golden_ratio, n) - np.power((1 - golden_ratio), n)) / np.sqrt(5))


def find_min(func: Callable[[float], float], interval: np.ndarray, inter_count: np.int64,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
    """Finds the minimum value of a function within a given interval using the Fibonacci method.

    Parameters:
        func (Callable[[float], float]): The function to minimize.
        interval (np.ndarray): The interval [start, end] within which to search for the minimum.
        inter_count (np.int64) : The count of subintervals to divide main interval each iteration
        eps (float): The precision of the search.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
        float: The minimum value of the function or the x value that minimizes the function.
    """
    start: float = interval[0]
    end: float = interval[1]

    iteration = 1

    for i in range(1, inter_count):
        x1 = start + (end - start) * fibonacci(inter_count - i - 1) / fibonacci(inter_count - i + 1)
        x2 = start + (end - start) * fibonacci(inter_count - i) / fibonacci(inter_count - i + 1)

        if func(x1) > func(x2):
            start = x1
        else:
            end = x2

        if verbose:
            print(
                f"Round: {iteration},\tInterval: [{start}, {end}],\tF(X1): {func(x1)},\tF(X2): {func(x2)}")
        iteration += 1
        if (end - start) <= eps:
            break

    if func(end) > func((start + end) / 2):
        end = (start + end) / 2
    else:
        start = (start + end) / 2
    if verbose:
        print(
            f"Round: {iteration},\tInterval: [{start}, {end}],\tF(X1): {func(start)},\tF(X2): {func(end)}")

    if return_argmin:
        return (start + end) / 2
    else:
        return func((start + end) / 2)


def find_max(func: Callable[[float], float], interval: np.ndarray, inter_count: np.int64,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
    """Finds the maximum value of a function within a given interval using the Fibonacci method.

    Parameters:
        func (Callable[[float], float]): The function to maximize.
        interval (np.ndarray): The interval [start, end] within which to search for the maximum.
        inter_count (np.int64) : The count of subintervals to divide main interval each iteration
        eps (float): The precision of the search.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

    Returns:
        float: The maximum value of the function or the x value that maximizes the function.
    """

    def rev_func(x: float) -> float:
        return (-1) * func(x)

    return find_min(rev_func, interval, inter_count, eps, verbose, return_argmin)
