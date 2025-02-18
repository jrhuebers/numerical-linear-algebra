"""
    Module linear_solver.py
    Date: 09.12.2019

    Contains a function solve_lu(pr, l, u, pc, b), which solve the linear system Ax = b
    via forward and backward substitution given the decomposition pr * A * pc = l * u.
    Also contains functions solve_gs(A, b, x0, params), solve_es(A, b, x0, params),
    which solve the same problem using the Jacobi- or Gauß-Seidel method.
"""


from numpy.linalg import inv
import numpy as np

from scipy.sparse import tril, triu
from scipy.sparse.linalg import spsolve, spsolve_triangular
import scipy.linalg
import scipy.sparse.linalg



def solve_lu(pr, l, u, pc, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition pr * A * pc = l * u.

    Parameters
    ----------
    pr : scipy.sparse.csr_matrix
        row permutation matrix of LU-decomposition
    l : scipy.sparse.csr_matrix
        lower triangular unit diagonal matrix of LU-decomposition
    u : scipy.sparse.csr_matrix
        upper triangular matrix of LU-decomposition
    pc : scipy.sparse.csr_matrix
        column permutation matrix of LU-decom

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """

    y1 = spsolve(pr.T, b)
    y2 = spsolve_triangular(l, y1, lower=True)
    y3 = spsolve_triangular(u, y2, lower=False)
    return spsolve(pc.T, y3)


def solve_gs(A, b, x0, params=dict(eps=1e-8, max_iter=1000, min_red=1e-4)):
    """ Solves the linear system Ax = b via the Jacobi method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray
        right-hand-side of the linear system
    x0 : numpy.ndarray
        initial guess of the solution
    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float
            minimal reduction of the residual in the infinity norm in every
            step. If set less or equal to 0 no constraint on the norm of the
            reduction of the residual is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list
        iterates of the algorithm. First entry is ‘x0‘.
    list
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., ‘eps=0‘ and ‘max_iter=0‘, etc.
    """

    if params["eps"] <= 0 and params["max_iter"] < 1 and params["min_red"] <= 0:
        raise ValueError

    D_inverse = np.diag(1/A.diagonal())

    d = D_inverse @ b
    B = -D_inverse @ (tril(A, -1) + triu(A, 1))

    sequence = [x0]
    residual_list = [A @ x0 - b]

    x_n = x0
    i = 0
    res_norm = scipy.linalg.norm(residual_list[0], np.inf)
    while True:
        if res_norm < params["eps"] > 0:
            return "eps", sequence, residual_list

        x_n = np.ravel(B @ x_n + d)
        residual = A @ x_n - b

        sequence.append(x_n)
        residual_list.append(residual)

        res_norm = scipy.linalg.norm(residual, np.inf)

        if i > 0:
            norm_diff = abs(res_norm - scipy.linalg.norm(residual_list[-2], np.inf))
            if norm_diff <= params["min_red"] > 0:
                return "min_red", sequence, residual_list

        i += 1
        if i == params["max_iter"] >= 1:
            return "max_iter", sequence, residual_list

def solve_es(A, b, x0, params=dict(eps=1e-8, max_iter=1000, min_red=1e-4)):
    """ Solves the linear system Ax = b via the Gauß-Seidel method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray
        right-hand-side of the linear system
    x0 : numpy.ndarray
        initial guess of the solution
    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float
            minimal reduction of the residual in the infinity norm in every
            step. If set less or equal to 0 no constraint on the norm of the
            reduction of the residual is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list
        iterates of the algorithm. First entry is ‘x0‘.
    list
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., ‘eps=0‘ and ‘max_iter=0‘, etc.
    """

    if params["eps"] <= 0 and params["max_iter"] < 1 and params["min_red"] <= 0:
        raise ValueError

    inverse = inv(tril(A, -1) + np.diag(A.diagonal()))

    B = -inverse @ triu(A, 1)
    d = inverse @ b

    sequence = [x0]
    residual_list = [A @ x0 - b]

    x_n = x0
    i = 0
    res_norm = scipy.linalg.norm(residual_list[0], np.inf)
    while True:
        if res_norm < params["eps"] > 0:
            return "eps", sequence, residual_list

        x_n = np.ravel(B @ x_n + d)
        residual = A @ x_n - b

        sequence.append(x_n)
        residual_list.append(residual)
        res_norm = scipy.linalg.norm(residual, np.inf)

        if i > 0:
            norm_diff = abs(res_norm - scipy.linalg.norm(residual_list[-2], np.inf))
            if norm_diff <= params["min_red"] > 0:
                return "min_red", sequence, residual_list

        i += 1
        if i == params["max_iter"] >= 1:
            return "max_iter", sequence, residual_list
