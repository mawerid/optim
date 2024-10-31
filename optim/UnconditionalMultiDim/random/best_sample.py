"""Best Sample Random Search method for unconstrained optimization.
"""

from typing import Callable

import numpy as np


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, start_step: float = 1.0, alpha: float = 0.7,
             sample_count: int = 10, eps: float = 1e-3, eps_func: float = 1e-5,
             max_iter: int = 1e5, verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the minimum value of a function within a given interval using the Best Sample Random Search method.

  Parameters:
      func (Callable[[np.ndarray], float]): The function to minimize.
      init_point (np.ndarray): The initial point to start optimization.
      ndim (int): The number of dimensions of the function.
      start_step (float): The initial value of step length.
      alpha (float): The coefficient of decrease of step length.
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
    raise ValueError("Initial point shape must be (ndim,).")

  point = np.array(init_point, dtype=float)
  previous_point = point + np.random.rand(ndim) * 10
  previous_point_func = func(previous_point)
  point_func = func(point)
  iteration = 0
  step_length = start_step

  while (np.linalg.norm(previous_point - point) > eps and
         np.linalg.norm(previous_point_func - point_func) > eps_func and
         iteration < max_iter and step_length > eps):
    if verbose:
      print(
        f"Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}")

    rand_vectors = np.random.uniform(-1, 1, (sample_count, ndim))
    rand_vectors /= np.linalg.norm(rand_vectors, axis=1)[:, None]
    rand_vectors *= step_length
    attempts = point + rand_vectors
    min_arg = np.argmin(np.apply_along_axis(func, 1, attempts))

    iteration += 1

    if func(attempts[min_arg]) >= point_func:
      step_length *= alpha
      continue

    previous_point = point.copy()
    point = attempts[min_arg]
    previous_point_func = point_func
    point_func = func(point)

  return point if return_argmin else np.array([func(point)])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, start_step: float = 1.0, alpha: float = 0.7,
             sample_count: int = 10, eps: float = 1e-3, eps_func: float = 1e-5,
             max_iter: int = 1e5, verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the maximum value of a function within a given interval using the Best Sample Random Search method.

  Parameters:
      func (Callable[[np.ndarray], float]): The function to maximise.
      init_point (np.ndarray): The initial point to start optimization.
      ndim (int): The number of dimensions of the function.
      start_step (float): The initial value of step length.
      alpha (float): The coefficient of decrease of step length.
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

  return find_min(rev_func, init_point, ndim, start_step, alpha, sample_count, eps, eps_func,
                  max_iter, verbose, return_argmin)
