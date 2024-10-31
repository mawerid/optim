"""Conjugate Gradient Descent method for unconstrained optimization.
"""

from typing import Callable

import numpy as np

from optim.utils.deriv import gradient
from optim.UnconditionalOneDim import newton


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 1e5,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the minimum value of a function within a given interval using Conjugate Gradient descent method.

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
  grad = gradient(func, point, ndim, eps)
  previous_grad = 0
  step = 0
  iteration = 0

  while (np.linalg.norm(previous_point - point) > eps and
         np.linalg.norm(previous_point_func - point_func) > eps_func and
         np.linalg.norm(grad) > eps_func and
         iteration < max_iter):
    if verbose:
      print(
        f"Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}")
    previous_point = point.copy()

    grad = gradient(func, point, ndim, eps)

    if verbose:
      print(
        f"Gradient in current point: {np.linalg.norm(grad)}")

    grad /= np.linalg.norm(grad)

    if iteration != 0:
      grad = grad + np.power(np.linalg.norm(grad) / np.linalg.norm(previous_grad), 2) * step

    step = grad

    def optim_func(a: float) -> float:
      return func(point - a * step)

    alpha = newton.find_min(optim_func, 0, eps=eps, max_iter=max_iter, verbose=verbose, return_argmin=True)

    point -= alpha * step

    iteration += 1
    previous_point_func = point_func
    point_func = func(point)
    previous_grad = grad

  return point if return_argmin else np.array([func(point)])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 1e5,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
  """Finds the maximum value of a function within a given interval using the Conjugate Gradient descent method.

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

  def rev_func(x: np.ndarray) -> float:
    return (-1) * func(x)

  return find_min(rev_func, init_point, ndim, eps, eps_func,
                  max_iter, verbose, return_argmin)
