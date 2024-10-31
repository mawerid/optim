"""Uniform search method for unconstrained optimization.
"""

from typing import Callable

import numpy as np


def find_min(func: Callable[[float], float], interval: np.ndarray, inter_count: np.int64,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
  """Finds the minimum value of a function within a given interval using the uniform search method.

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
  vec_func = np.vectorize(func)

  while np.abs(start - end) >= eps:
    nodes = np.linspace(start, end, num=inter_count, dtype=float)
    min_k = np.argmin(vec_func(nodes))
    if min_k == inter_count:
      min_k -= 1
    elif min_k == 0:
      min_k += 1
    start, end = nodes[min_k - 1], nodes[min_k + 1]

    if verbose:
      print(
        f"Round: {iteration},\tInterval: [{start}, {end}],\tK of minimal F value: {min_k},\tF(K): {func(min_k)}")
    iteration += 1
  if return_argmin:
    return (start + end) / 2
  else:
    return func((start + end) / 2)


def find_max(func: Callable[[float], float], interval: np.ndarray, inter_count: np.int64,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
  """Finds the maximum value of a function within a given interval using the uniform search method.

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
