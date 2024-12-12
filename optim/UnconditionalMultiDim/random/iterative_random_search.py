"""Iterative Random Search method for unconstrained optimization."""

from typing import Callable

import numpy as np


def find_min(
    func: Callable[[np.ndarray], float],
    init_point: np.ndarray,
    ndim: int = 1,
    start_step: float = 1.0,
    alpha: float = 0.75,
    betta: float = 0.7,
    gamma: float = 0.7,
    sample_count: int = 10,
    eps: float = 1e-3,
    eps_func: float = 1e-5,
    max_iter: int = 1e5,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the minimum value of a function within a given interval using the Iterative Random Search method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to minimize.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        start_step (float): The initial value of step length.
        alpha (float): The coefficient of decrease of step length.
        betta (float): The weight of the previous vector component and the random component.
        gamma (float): The weight of the previous vector component and prev previous vector component.
        sample_count (int): The number of samples to generate.
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

    point = np.array(init_point, dtype=float)
    previous_point = point.copy() + np.random.rand(ndim) * 10

    prev_component = np.zeros(ndim)
    prev_prev_component = np.zeros(ndim)
    previous_point_func = func(previous_point)
    point_func = func(point)
    iteration = 0
    step_length = start_step

    # First two iterations for initialization
    for _ in range(2):
        if verbose:
            print(
                f'Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}'
            )

        rand_vectors = np.random.uniform(-1, 1, (sample_count, ndim))
        norm_rand_vectors = rand_vectors / np.linalg.norm(rand_vectors, axis=1)[:, None]
        norm_rand_vectors *= step_length
        attempts = point + norm_rand_vectors
        min_arg = np.argmin(np.apply_along_axis(func, 1, attempts))

        iteration += 1

        if func(attempts[min_arg]) >= point_func:
            step_length *= alpha
            continue

        prev_prev_component = prev_component.copy()
        prev_component = rand_vectors[min_arg].copy()

        previous_point = point.copy()
        point = attempts[min_arg]
        previous_point_func = point_func
        point_func = func(point)

    step_length = start_step

    # Main loop
    while (
        np.linalg.norm(previous_point - point) > eps
        and np.linalg.norm(previous_point_func - point_func) > eps_func
        and iteration < max_iter
        and step_length > eps
    ):
        if verbose:
            print(
                f'Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}'
            )

        component = gamma * prev_component + (1 - gamma) * prev_prev_component
        norm_component = component / np.linalg.norm(component)
        attempt = point.copy()
        delta = 0
        k = 0

        while func(attempt) >= point_func and k < sample_count:
            rand_vector = np.random.uniform(0, 1, ndim)
            rand_vector /= np.linalg.norm(rand_vector)

            delta = betta * norm_component + (1 - betta) * rand_vector
            attempt = point + delta * step_length
            k += 1

        iteration += 1

        if func(attempt) >= point_func:
            step_length *= alpha
            print(step_length)
            continue

        step_length /= alpha

        prev_prev_component = prev_component.copy()
        prev_component = delta

        previous_point = point.copy()
        point = attempt
        previous_point_func = point_func
        point_func = func(point)

    return point if return_argmin else np.array([func(point)])


def find_max(
    func: Callable[[np.ndarray], float],
    init_point: np.ndarray,
    ndim: int = 1,
    start_step: float = 1.0,
    alpha: float = 0.7,
    betta: float = 0.7,
    gamma: float = 0.7,
    sample_count: int = 10,
    eps: float = 1e-3,
    eps_func: float = 1e-5,
    max_iter: int = 1e5,
    verbose: bool = False,
    return_argmin: bool = True,
) -> np.ndarray:
    """Finds the maximum value of a function within a given interval using the Iterative Random Search method.

    Parameters:
        func (Callable[[np.ndarray], float]): The function to maximise.
        init_point (np.ndarray): The initial point to start optimization.
        ndim (int): The number of dimensions of the function.
        start_step (float): The initial value of step length.
        alpha (float): The coefficient of decrease of step length.
        betta (float): The weight of the previous vector component and the random component.
        gamma (float): The weight of the previous vector component and prev previous vector component.
        sample_count (int): The number of samples to generate.
        eps (float): The precision of the search.
        eps_func (float): The precision of the function value.
        max_iter (int): The maximum number of iterations to perform.
        verbose (bool): If True, prints detailed information about each iteration.
        return_argmin (bool): If True, returns the x value that maximise the function. Otherwise, returns the minimum function value.

    Returns:
        np.ndarray: The minimum value of the function or the x value that minimizes the function.
    """

    def rev_func(x: np.ndarray) -> float:
        return (-1) * func(x)

    return find_min(
        rev_func,
        init_point,
        ndim,
        start_step,
        alpha,
        betta,
        gamma,
        sample_count,
        eps,
        eps_func,
        max_iter,
        verbose,
        return_argmin,
    )
