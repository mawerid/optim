from typing import Callable

import numpy as np

from optim.utils.deriv import gradient
from optim.UnconditionalOneDim import newton


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 100,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the minimum value of a function within a given interval using Steepest Gradient descent method.

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
    if init_point.shape != (ndim,):
        raise ValueError("Initial point shape must be (ndim,).")

    point = np.array(init_point, dtype=float)
    previous_point = point + np.random.rand(ndim) * 10
    previous_point_func = func(previous_point)
    point_func = func(point)
    iteration = 0
    grad = gradient(func, point, ndim, eps if eps < 1e-5 else 1e-5)

    while (np.linalg.norm(previous_point - point) > eps and
           np.linalg.norm(previous_point_func - point_func) > eps_func and
           np.linalg.norm(grad) > eps_func and
           iteration < max_iter):
        if verbose:
            print(
                f"Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}")
        previous_point = point.copy()

        grad = gradient(func, point, ndim, eps if eps < 1e-5 else 1e-5)

        if verbose:
            print(
                f"Gradient in current point: {np.linalg.norm(grad)}")

        grad /= np.linalg.norm(grad)

        optim_func = lambda a: func(point - a * grad)
        alpha = newton.find_min(optim_func, 0, eps=eps, max_iter=max_iter, verbose=verbose, return_argmin=True)

        point -= alpha * grad

        iteration += 1
        previous_point_func = point_func
        point_func = func(point)

    return point if return_argmin else np.array([func(point)])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 1e2,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """Finds the maximum value of a function within a given interval using the Steepest Gradient descent method.

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
