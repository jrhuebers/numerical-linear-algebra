"""
    Module linear_solver.py
    Date: 09.12.2019

    Contains a function solve_lu(pr, l, u, pc, b) which solves the linear system Ax = b
    via forward and backward substitution given the decomposition pr * A * pc = l * u.
"""


from scipy.sparse.linalg import spsolve, spsolve_triangular



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
