from typing import Callable
import numpy as np
from optim.UnconditionalOneDim import newton


def ortogonalization(vectors: np.ndarray) -> np.ndarray:
    """Orthogonalizes a set of n-dimensional vectors using the Gram-Schmidt process.

        Args:
            vectors (np.ndarray): A 2D array where each row represents a vector to be orthogonalized.

        Returns:
            np.ndarray: A 2D array where each row is an orthogonalized vector.
        """
    if vectors.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if vectors.shape[0] < 1:
        raise ValueError("No vectors provided for orthogonalization.")

    rank = np.linalg.matrix_rank(vectors)
    if rank < vectors.shape[0]:
        raise ValueError("Vectors are linearly dependent.")

    orthogonal_vectors = np.zeros_like(vectors, dtype=float)
    orthogonal_vectors[0] = vectors[0] / np.linalg.norm(vectors[0])
    for i in range(1, vectors.shape[0]):
        projections = np.dot(orthogonal_vectors[:i], vectors[i]) / np.linalg.norm(orthogonal_vectors[:i], axis=1) ** 2
        projections = np.dot(projections[:, np.newaxis], orthogonal_vectors[:i]).sum(axis=0)
        orthogonal_vec = vectors[i] - projections
        orthogonal_vectors[i] = orthogonal_vec / np.linalg.norm(orthogonal_vec)

    return orthogonal_vectors


def find_min(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 100,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """
    Finds the minimum value of a function within a given interval using the Gauss-Seidel method.

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

    while (np.linalg.norm(previous_point - point) > eps and
           np.linalg.norm(previous_point_func - point_func) > eps_func and
           iteration < max_iter):
        if verbose:
            print(
                f"Iteration: {iteration}, previous point: {previous_point}, current point: {point}, F(current point): {func(point)}")

        previous_point = point.copy()

        new_coords = np.zeros((ndim, ndim), dtype=float)

        for dim in range(ndim):
            one_dim_func = lambda x, d=dim: func(np.array([val if i != d else x for i, val in enumerate(point)]))
            if verbose:
                print(f"Starting dimension {dim} optimization")
            point[dim] = newton.find_min(one_dim_func, float(point[dim]), eps, max_iter, verbose=verbose,
                                         return_argmin=True)
            new_coords[dim] = point

        print(new_coords)
        print(ortogonalization(new_coords))
        point = ortogonalization(new_coords) @ point

        iteration += 1
        previous_point_func = point_func
        point_func = func(point)

    return point if return_argmin else np.array([func(point)])


def find_max(func: Callable[[np.ndarray], float], init_point: np.ndarray,
             ndim: int = 1, eps: float = 1e-3, eps_func: float = 1e-5, max_iter: int = 1e2,
             verbose: bool = False, return_argmin: bool = True) -> np.ndarray:
    """
    Finds the maximum value of a function within a given interval using the Gauss-Seidel method.

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