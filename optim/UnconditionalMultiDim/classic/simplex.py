"""Simplex method for unconstrained optimization.
"""

from typing import Callable

import numpy as np

from optim.utils import simplex as simp


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, length: float = 10.0, split_length: float = 0.7, eps: float = 1e-3, max_iter: int = 1e5,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the minimum value of a function using the Simplex method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        length (float): The length of the simplex edge.
        split_length (float): The length of the simplex edge after a reflection.
        eps (float): The precision of the search.
        max_iter (int): The maximum number of iterations to perform.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
        np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """
    if init_point.shape != (ndim,):
        raise ValueError("Initial point shape must be (ndim,).")

    point = np.array(init_point, dtype=float)
    iteration = 0

    best_idx = 0
    simplex = simp.generate(point, ndim, length)
    previous_simplex = simplex.copy() + 100 * eps
    reflected = point + np.random.rand(ndim) * 10

    while (np.linalg.norm(reflected - simplex[best_idx]) > eps and
           iteration < max_iter and length > eps):
        prev_prev_simplex = previous_simplex.copy()
        previous_simplex = simplex.copy()
        func_simplex = np.apply_along_axis(func, 1, simplex)
        worst_idx = np.argmax(func_simplex)
        best_idx = np.argmin(func_simplex)

        simplex = simp.reflect(simplex, worst_idx)

        if np.linalg.norm(prev_prev_simplex - simplex) < eps:
            length *= split_length
            simplex = simp.reduce(simplex, best_idx, split_length)

        if verbose:
            print(f"Iteration: {iteration}, simplex: {simplex}, F(simplex): {np.apply_along_axis(func, 1, simplex)}")

        iteration += 1

    return simplex[best_idx] if return_argmin else np.array([func(simplex[best_idx])])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, length: float = 10.0, split_length: float = 0.7, eps: float = 1e-3, max_iter: int = 1e5,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the maximum value of a function using the Simplex method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to maximize.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        length (float): The length of the simplex edge.
        split_length (float): The length of the simplex edge after a reflection.
        eps (float): The precision of the search.
        max_iter (int): The maximum number of iterations to perform.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

    Returns:
        np.ndarray: The maximum value of the function or the x value that maximizes the function.
    """

    def rev_func(x: np.ndarray) -> float:
        return (-1) * func(x)

    return find_min(rev_func, init_point, ndim, length, split_length, eps, max_iter, verbose, return_argmin)
