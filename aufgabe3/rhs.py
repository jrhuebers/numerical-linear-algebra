"""
    Module rhs.py:
    Contains funcion rhs for computing the right-hand side vector `b` for a
    given function `f` for Approximation of a Solution to the Poisson problem.

    Date: 24.11.2019
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python


import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import block_matrix
import linear_solvers


def rhs(d, n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is a scalar or array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.

    """

    if (d < 1) or (n < 2):
        raise ValueError

    # pylint: disable=cell-var-from-loop
    pt_array = np.array([np.vectorize(lambda l: (int(m/(n-1)**l)%(n-1)+1)/n)(range(d))
                         for m in range((n-1)**d)])

    b = np.array([float(f(x)/(n**2)) for x in pt_array])
    return b

def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the max-norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of ’numpy’
        Finite difference approximation of the solution of the Poisson problem
        at the disretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is `u(x)`. Here `x` is a scalar
        or array_like of `numpy`. The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """

    b = rhs(d, n, u)

    return abs(hat_u - b*n**2).max()

def plot_error(d, u, laplace_u, n_start=2, n_end=10):
    """ Produces a plot of the error (max-norm) of the numerical solution
    of the Poisson problem with respect to N.

    Parameters
    ----------
    d : int
        Dimension of the space.
    u : callable
        Solution of the Poisson problem
        The calling signature is `u(x)`. Here `x` is a scalar
        or array_like of `numpy`. The return value is a scalar.
    laplace_u : callable
        Laplace-Operator for function u
        The calling signature in `laplace_u(x)`. Here `x` is a scalar
        or array_like of `numpy`. The return value is a scalar.
    n_start : int, optional
        First plotpoints Number of intervals
    n_end : int, optional
        Last plotpoints Number of intervals
    """

    n_list = list(range(n_start, n_end+1))
    error_list = list()
    for n in n_list:
        matrix = block_matrix.BlockMatrix(d, n)
        pr, l_matrix, u_matrix, pc = matrix.get_lu()

        f = lambda x: -laplace_u(x)
        hat_u = linear_solvers.solve_lu(pr, l_matrix, u_matrix, pc, rhs(d, n, f))

        error_list.append(compute_error(d, n, hat_u, u))

    plt.plot(n_list, error_list)

    plt.title("Maximal absolute error with respect to N")
    plt.xlabel("N")
    plt.ylabel("max. abs. error")
    plt.show()
