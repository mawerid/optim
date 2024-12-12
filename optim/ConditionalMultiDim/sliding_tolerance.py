"""Sliding tolerance method for constrained optimization."""

from typing import List, Callable

import numpy as np


def find_min(
    func: Callable[[np.ndarray], float],
    init_point: np.ndarray,
    comp_constrains: List[Callable[[np.ndarray], float]] = None,
    comp_coeff: np.ndarray = None,
    ndim: int = 1,
    eps: float = 1e-3,
    eps_func: float = 1e-5,
    max_iter: int = 1e5,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the minimum value of a function within a given interval with conditional constrains using the Sliding tolerance method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.
        init_point (np.ndarray): The initial point to start optimization.
        equ_constrains (List[Callable[[np.ndarray], float]]): The equality constrains of the function.
        comp_constrains (List[Callable[[np.ndarray], float]]): The inequality constrains of the function.
        equ_coeff (np.ndarray): The coefficients for the equality constrains.
        comp_coeff (np.ndarray): The coefficients for the inequality constrains.
        alphas (np.ndarray): The coefficients for the penalty functions.
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
        raise ValueError('Initial point shape must be (ndim,).')

    if comp_constrains is None:
        comp_constrains = []
    if comp_coeff is None:
        comp_coeff = np.ones(len(comp_constrains))

    def tolerance_function(
        point: np.ndarray,
        comp_coeff: np.ndarray,
    ) -> float:
        return np.sum(
            comp_coeff
            * np.maximum(0, np.array([constr(point) for constr in comp_constrains])) ** 2
        )

    def tolerance_criterion():
        pass

    def check_constrains(point: np.ndarray) -> bool:
        comp_values = np.array([constraint(point) for constraint in comp_constrains])
        return np.all(comp_values >= -eps)

    point = init_point.copy()
    # previous_point = point + np.random.rand(ndim) * 10
    # previous_point_func = func(previous_point)
    # point_func = func(point)
    # iteration = 0
    # prev_point_index = 0

    # while (
    #     np.linalg.norm(previous_point - point) > eps
    #     and np.linalg.norm(previous_point_func - point_func) > eps_func
    #     and iteration < max_iter
    # ):
    #     if verbose:
    #         print(
    #             f'Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}, Penalty: {penalty_func(point, alphas, equ_coeff, comp_coeff)}, Prev Point Index: {prev_point_index}'
    #         )

    #     new_point = conjugate_descent.find_min(
    #         lambda x: penalty_func(x, alphas, equ_coeff, comp_coeff),
    #         point,
    #         ndim,
    #         eps,
    #         eps_func,
    #         max_iter,
    #         verbose,
    #     )
    #     if check_constrains(new_point):
    #         if iteration // 2 == prev_point_index:
    #             previous_point = point.copy()
    #             previous_point_func = point_func
    #             prev_point_index += 1
    #         point = new_point
    #         point_func = func(point)
    #     else:
    #         if verbose:
    #             print('Constrains are not satisfied.')
    #         alphas *= 1.1

    #     iteration += 1

    if return_argmin:
        return point
    return func(point)


def find_max(
    func: Callable[[np.ndarray], float],
    init_point: np.ndarray,
    equ_constrains: List[Callable[[np.ndarray], float]] = None,
    comp_constrains: List[Callable[[np.ndarray], float]] = None,
    equ_coeff: np.ndarray = None,
    comp_coeff: np.ndarray = None,
    alphas: np.ndarray = None,
    ndim: int = 1,
    eps: float = 1e-3,
    eps_func: float = 1e-5,
    max_iter: int = 1e5,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the maximum value of a function within a given interval with conditional constrains using the Sliding tolerance method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.
        init_point (np.ndarray): The initial point to start optimization.
        equ_constrains (List[Callable[[np.ndarray], float]]): The equality constrains of the function.
        comp_constrains (List[Callable[[np.ndarray], float]]): The inequality constrains of the function.
        equ_coeff (np.ndarray): The coefficients for the equality constrains.
        comp_coeff (np.ndarray): The coefficients for the inequality constrains.
        alphas (np.ndarray): The coefficients for the penalty functions.
        ndim (int): The number of dimensions of the function.
        eps (float): The precision of the search.
        eps_func (float): The precision of the function value.
        max_iter (int): The maximum number of iterations to perform.
        verbose (bool): If True, prints detailed information about each iteration.

    Returns:
        np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """

    def rev_func(x: np.ndarray) -> float:
        return (-1) * func(x)

    return find_min(
        rev_func,
        init_point,
        equ_constrains,
        comp_constrains,
        equ_coeff,
        comp_coeff,
        alphas,
        ndim,
        eps,
        eps_func,
        max_iter,
        verbose,
        return_argmin,
    )
