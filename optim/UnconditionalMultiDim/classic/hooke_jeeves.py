"""Hooke-Jeeves method for unconstrained optimization.
"""

from typing import Callable

import numpy as np

from optim.UnconditionalOneDim import newton


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray, alpha: float = 1.0,
             initial_trial_vector: np.ndarray = None, ndim: int = 1, eps: float = 1e-3,
             eps_func: float = 1e-5, max_iter: int = 1e5, trial_vector_decrease: float = 0.5,
             alpha_type: str = 'c', verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the minimum value of a function using the Hooke-Jeeves method.

  Parameters:
      func (Callable[[np.ndarray], float]): The function to minimize.
      init_point (np.ndarray): The initial point to start optimization.
      alpha (float): The step size for the search.
      initial_trial_vector (np.ndarray): The initial trial vector to start optimization.
      ndim (int): The number of dimensions of the function.
      eps (float): The precision of the search.
      eps_func (float): The precision of the function value.
      max_iter (int): The maximum number of iterations to perform.
      trial_vector_decrease (float): The factor by which to decrease the step size.
      alpha_type (str): The type of step size to use. Can be 'c' for constant or 'd' for dynamic or 'o' for dynamic using optimization.
      verbose (bool): If True, prints detailed information about each iteration.
      return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

  Returns:
      np.ndarray: The minimum value of the function or the x value that minimizes the function.
  """
  if init_point.shape != (ndim,):
    raise ValueError("Initial point shape must be (ndim,).")

  point = np.array(init_point, dtype=float)
  current_alpha = alpha
  trial_vector = initial_trial_vector.copy() if initial_trial_vector is not None else np.ones(ndim) * 10

  if trial_vector.shape != (ndim,):
    raise ValueError("Initial trial vector shape must be (ndim,).")

  iteration = 0
  previous_point = point + np.random.rand(ndim) * 10
  previous_point_func = func(previous_point)
  point_func = func(point)

  while (np.linalg.norm(previous_point - point) > eps and
         np.linalg.norm(previous_point_func - point_func) > eps_func and
         iteration < max_iter):

    prev_prev_point = previous_point.copy()
    previous_trial_vector = trial_vector.copy()

    if verbose:
      print(
        f"Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}")

    previous_point = point.copy()
    real_trial_vector = np.zeros(ndim)

    for dim in range(ndim):

      def one_dim_func(x: float) -> float:
        p = point + real_trial_vector
        p[dim] = x
        return func(p)

      if one_dim_func(float(point[dim] + trial_vector[dim])) > one_dim_func(float(point[dim])):
        if one_dim_func(float(point[dim] - trial_vector[dim])) > one_dim_func(float(point[dim])):
          trial_vector[dim] = 0.0
        else:
          trial_vector[dim] *= -1.0
      real_trial_vector[dim] = trial_vector[dim]

    if alpha_type == 'd':
      # TODO: Implement dynamic step size
      pass
    elif alpha_type == 'o':
      current_alpha = newton.find_min(lambda x: func(point + x * trial_vector), current_alpha, eps, max_iter,
                                      verbose=verbose, return_argmin=True)

    point += current_alpha * trial_vector

    trial_vector = previous_trial_vector
    if np.linalg.norm(previous_point - point) < eps and np.linalg.norm(trial_vector) > eps:
      trial_vector *= trial_vector_decrease
      previous_point = prev_prev_point

    iteration += 1
    previous_point_func = func(previous_point)
    point_func = func(point)

  return point if return_argmin else np.array([func(point)])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray, alpha: float = 1.0,
             initial_trial_vector: np.ndarray = None, ndim: int = 1, eps: float = 1e-3,
             eps_func: float = 1e-5, max_iter: int = 1e5, trial_vector_decrease: float = 0.5,
             alpha_type: str = 'c', verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the maximum value of a function using the Hooke-Jeeves method.

  Parameters:
      func (Callable[[np.ndarray], float]): The function to maximize.
      init_point (np.ndarray): The initial point to start optimization.
      alpha (float): The step size for the search.
      initial_trial_vector (np.ndarray): The initial trial vector to start optimization.
      ndim (int): The number of dimensions of the function.
      eps (float): The precision of the search.
      eps_func (float): The precision of the function value.
      max_iter (int): The maximum number of iterations to perform.
      trial_vector_decrease (float): The factor by which to decrease the step size.
      alpha_type (str): The type of step size to use. Can be 'c' for constant or 'd' for dynamic or 'o' for dynamic using optimization.
      verbose (bool): If True, prints detailed information about each iteration.
      return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

  Returns:
      np.ndarray: The maximum value of the function or the x value that maximizes the function.
  """

  def rev_func(x: np.ndarray) -> float:
    return (-1) * func(x)

  return find_min(rev_func, init_point, alpha, initial_trial_vector, ndim, eps, eps_func, max_iter,
                  trial_vector_decrease, alpha_type, verbose, return_argmin)
