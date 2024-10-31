"""Simplex module.
"""

import numpy as np


def generate(init_point: np.ndarray, ndim: int, length: float = 1.0) -> np.ndarray:
  """Generates a simplex from an initial point.

  Parameters:
      init_point (np.ndarray): The initial point to start optimization.
      ndim (int): The number of dimensions of the function.
      length (float): The length of the simplex edge.

  Returns:
      np.ndarray: The simplex generated from the initial point.
  """
  if init_point.shape != (ndim,):
    raise ValueError("Initial point shape must be (ndim,).")

  simplex = np.zeros((ndim + 1, ndim))
  simplex[0] = init_point

  r_1 = length * (np.sqrt(ndim + 1) + ndim - 1) / (ndim * np.sqrt(2))
  r_2 = length * (np.sqrt(ndim + 1) - 1) / (ndim * np.sqrt(2))

  for i in range(1, ndim + 1):
    x = np.copy(init_point) + r_2
    x[i - 1] += r_1 - r_2
    simplex[i] = x

  return simplex


def reflect(simplex: np.ndarray, point_id: int) -> np.ndarray:
  """Reflects the worst point of the simplex.

  Parameters:
      simplex (np.ndarray): The simplex to reflect.
      point_id (int): The index of the point to reflect.

  Returns:
      np.ndarray: The reflected simplex.
  """
  centroid = np.mean(np.delete(simplex, point_id, axis=0), axis=0)
  reflected = 2 * centroid - simplex[point_id]

  simplex[point_id] = reflected

  return simplex


def reduce(simplex: np.ndarray, point_id: int, length: float = 1.0) -> np.ndarray:
  """Reduces the simplex edge length.

  Parameters:
      simplex (np.ndarray): The simplex to reduce.
      point_id (int): The index of the point to reduce.
      length (float): The length of the simplex edge.

  Returns:
      np.ndarray: The reduced simplex.
  """
  for i, point in enumerate(simplex):
    if i != point_id:
      point += length * (simplex[point_id] - point)

  return simplex


def squeeze(simplex: np.ndarray, length: float = 1.0) -> np.ndarray:
  """Squeezes the simplex.

  Parameters:
      simplex (np.ndarray): The simplex to squeeze.
      length (float): The length of the simplex edge.

  Returns:
      np.ndarray: The squeezed simplex.
  """
  centroid = np.mean(simplex, axis=0)
  simplex = centroid + length * (simplex - centroid)

  return simplex
