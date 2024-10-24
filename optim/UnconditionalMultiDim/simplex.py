from typing import Callable

import numpy as np


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, l: float = 1.0, eps: float = 1e-5, max_iter: int = 1000,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the minimum value of a function using the Simplex method.

    Parameters:
    func (Callable[[np.ndarray], float]): The function to minimize.
    init_point (np.ndarray): The initial point to start optimization.
    ndim (int): The number of dimensions of the function.
    l (float): The length of the simplex edge.
    eps (float): The precision of the search.
    max_iter (int): The maximum number of iterations to perform.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """
    if init_point.shape != (ndim,):
        raise ValueError("Initial point shape must be (ndim,).")

    n = ndim
    simplex = np.zeros((n + 1, n))
    simplex[0] = init_point

    for i in range(1, n + 1):
        x = np.copy(init_point)
        x[i - 1] += l
        simplex[i] = x

    for iteration in range(max_iter):
        f_values = np.apply_along_axis(func, 1, simplex)
        worst_idx = np.argmax(f_values)
        best_idx = np.argmin(f_values)

        centroid = np.mean(np.delete(simplex, worst_idx, axis=0), axis=0)
        reflected = centroid + (centroid - simplex[worst_idx])

        if np.linalg.norm(reflected - simplex[best_idx]) < eps:
            break

        f_reflected = func(reflected)
        if f_reflected < f_values[worst_idx]:
            simplex[worst_idx] = reflected

        if verbose:
            print(f"Iteration: {iteration}, simplex: {simplex}, F(simplex): {f_values}")

    return simplex[best_idx] if return_argmin else np.array([func(simplex[best_idx])])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 1e2,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the maximum value of a function within a given interval using the Simplex method.

    Parameters:
    func (Callable[[np.ndarray], float]): The function to minimize.
    init_point (np.ndarray): The initial point to start optimization.
    ndim (int): The number of dimensions of the function.
    eps (float): The precision of the search.
    eps_func (float): The precision of the function value.
    max_iter (int): The maximum number of iterations to perform.
    verbose (bool): If True, prints detailed information about each iteration.
    return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
    np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """
    rev_func = lambda x: -func(x)
    return find_min(rev_func, init_point, ndim, eps, eps_func,
                    max_iter, verbose, return_argmin)
