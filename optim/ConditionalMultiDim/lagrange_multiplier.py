"""Lagrange Multiplier method for constrained optimization."""

from typing import List, Callable

import numpy as np

from optim.UnconditionalMultiDim.gradient import steepest_gradient_descent


def find_min(
    func: Callable[[np.ndarray], float],
    comp_constrains: List[Callable[[np.ndarray], float]],
    init_point: np.ndarray,
    ndim: int = 1,
    eps: float = 1e-3,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the minimum value of a function within a given interval with conditional constrains using the Lagrange Multiplier method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.        comp_constrains (List[Callable[[np.ndarray], float]]): The constrains of the function. (all constrains must be via g(x) > 0)
        equ_lambdas (np.ndarray): The Lagrange Multipliers for equality constrains.
        comp_lambdas (np.ndarray): The Lagrange Multipliers for constrains.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        eps (float): The precision of the search.
        eps_func (float): The precision of the function value.
        max_iter (int): The maximum number of iterations to perform.
        alpha (float): The initial step len coefficient.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
        np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """
    if comp_constrains is None:
        comp_constrains = []

    def lagrange(point: np.ndarray, comp_lambdas: np.ndarray) -> np.ndarray:
        full_func = func(point)
        for i, constraint in enumerate(comp_constrains):
            full_func += comp_lambdas[i] * constraint(point)
        return full_func

    def check_constrains(point: np.ndarray) -> bool:
        for constraint in comp_constrains:
            if constraint(point) < -eps:
                return False
        return True

    point = init_point.copy()
    constrain_size = len(comp_constrains)
    points_num = 2**constrain_size
    principal_points = np.zeros((points_num, ndim))

    for i in range(points_num):
        lambdas = np.zeros(constrain_size)
        for j in range(constrain_size):
            lambdas[j] = 1 if i & (1 << j) else 0

        principal_points[i] = steepest_gradient_descent.find_min(
            lambda x: lagrange(x, lambdas),
            point,
            ndim=ndim,
            eps=eps,
            verbose=verbose,
        )

    print(principal_points)
    filtered_points = np.apply_along_axis(check_constrains, 1, principal_points)
    print(filtered_points)
    point = principal_points[
        np.apply_along_axis(func, 1, principal_points[filtered_points]).argmin()
    ]

    if return_argmin:
        return point
    return func(point)


def find_max(
    func: Callable[[np.ndarray], float],
    constrains: List[Callable[[np.ndarray], float]],
    lambdas: np.ndarray,
    init_point: np.ndarray,
    ndim: int = 1,
    eps: float = 1e-3,
    eps_func: float = 1e-5,
    max_iter: int = 1e5,
    alpha: float = 1e-2,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the maximum value of a function within a given interval with conditional constrains using the Lagrange Multiplier method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.
        constrains (List[Callable[[np.ndarray], float]]): The constrains of the function. (all constrains must be via g(x) >= 0)
        lambdas (np.ndarray): The Lagrange Multipliers.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        eps (float): The precision of the search.
        eps_func (float): The precision of the function value.
        max_iter (int): The maximum number of iterations to perform.
        alpha (float): The initial step len coefficient.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

    Returns:
        np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """

    def rev_func(x: np.ndarray) -> float:
        return (-1) * func(x)

    return find_min(
        rev_func,
        constrains,
        lambdas,
        init_point,
        ndim,
        eps,
        eps_func,
        max_iter,
        alpha,
        verbose,
        return_argmin,
    )
