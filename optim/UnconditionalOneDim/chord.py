"""Chord method for unconstrained optimization.
"""

from typing import Callable

import numpy as np

from optim.utils.deriv import derivative


def find_min(func: Callable[[float], float], interval: np.ndarray,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
  """Finds the minimum value of a function within a given interval using the chord method.

  Parameters:
      func (Callable[[float], float]): The function to minimize.
      interval (np.ndarray): The interval [start, end] within which to search for the minimum.
      eps (float): The precision of the search.
      verbose (bool): If True, prints detailed information about each iteration.
      return_argmin (bool): If True, returns the x value that minimizes the function. Otherwise, returns the minimum function value.

  Returns:
      float: The minimum value of the function or the x value that minimizes the function.
  """
  start: float = interval[0]
  end: float = interval[1]

  x = (end - start) / 2
  x_derivative = derivative(func, x, eps)
  iteration = 1

  while np.abs(x_derivative) >= eps:
    start_derivarive = derivative(func, start, eps)
    end_derivarive = derivative(func, end, eps)
    if start_derivarive * end_derivarive > 0:
      raise ValueError("The function has no minimum within the given interval.")
    x = (end - (end - start) * end_derivarive
         / (end_derivarive - start_derivarive))
    x_derivative = derivative(func, x, eps)
    if start_derivarive * x_derivative < 0:
      end = x
    elif x_derivative * end_derivarive < 0:
      start = x
    if verbose:
      print(
        f"Round: {iteration},\tInterval: [{start}, {end}],\tx: {x},\tF'(x): {x_derivative}")
    iteration += 1

  return x if return_argmin else func(x)


def find_max(func: Callable[[float], float], interval: np.ndarray,
             eps: float = 1e-3, verbose: bool = False, return_argmin: bool = False) -> float:
  """Finds the maximum value of a function within a given interval using the chord method.

  Parameters:
      func (Callable[[float], float]): The function to maximize.
      interval (np.ndarray): The interval [start, end] within which to search for the maximum.
      eps (float): The precision of the search.
      verbose (bool): If True, prints detailed information about each iteration.
      return_argmin (bool): If True, returns the x value that maximizes the function. Otherwise, returns the maximum function value.

  Returns:
      float: The maximum value of the function or the x value that maximizes the function.
  """

  def rev_func(x: float) -> float:
    return (-1) * func(x)

  return find_min(rev_func, interval, eps, verbose, return_argmin)
