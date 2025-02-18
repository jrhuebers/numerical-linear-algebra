"""
    Module main.py
    Date: 09.12.2019

    Implements a main-function to test the methods related to approximation of the Poisson problem
    in modules block_matrix, linear_solvers, rhs. The solution u and the function laplace_u defining
    the relating Poisson problem

    [-Delta u(x) = f(x) := -laplace_u(x),  for x in (0,1)^d]

    are implemented as well.
"""

import block_matrix
import linear_solvers
import rhs


def u(x):
    """ Solution to the Poisson-Problem with laplace_u.
    u(x) = product over l of x_l*(1-x_l)

    Parameters
    ----------
    x : array_like of `numpy`
        Point in the area omega.

    Returns
    -------
    float
        value of function u at point x
    """

    product = 1
    for x_l in x:
        product *= x_l*(1-x_l)
    return product

def laplace_u(x):
    """ Laplace operator of function u.
    laplace_u(x) = sum over l of
                   [(product over k of x_k*(1-x_k)) * (1-2*x_l) * (product over k of x_k*(1-x_k))]

    Parameters
    ----------
    x : array_like of `numpy`
        Point in the area omega.

    Returns
    -------
    float
        value of laplace operator of u at point x
    """

    d = x.shape[0]
    sum = 0
    for l in range(d):
        d_l = 1
        for k in range(l):
            d_l *= x[k]*(1-x[k])
        d_l *= (1-2*x[l])
        for k in range(l+1, d):
            d_l *= x[k]*(1-x[k])

        sum += d_l

    return sum


def main():
    """ main-function of main.py to test methods from modules block_matrix, linear_solvers, rhs.
    Approximates solution u to the Laplace Problem with f=-laplace_u. Prints the maximal absolute
    error of this approximation for d=3, N=10, thus testing methods
    block_matrix.BlockMatrix.get_lu(), linear_solvers.solve_lu and rhs.compute_error.

    Then the maximal absolute errors of approximation for this Poisson problem are plotted against
    N, testing methods rhs.plot_error, rhs.compute_error, block_matrix.BlockMatrix.get_lu and
    linear_solvers.solve_lu.

    The plot of the condition of A(3) with respect to N tests methods block_matrix.plot_condition
    and block_matrix.BlockMatrix.get_cond.

    The plot of number of non-zero entries in matrices A(3) and L, U tests methods
    block_matrix.plot_non_zero and block_matrix.BlockMatrix.eval_zeros_lu.

    Thus all implemented methods are tested.
    """

    d = 3
    n = 10
    matrix = block_matrix.BlockMatrix(d, n)
    pr, lower, upper, pc = matrix.get_lu()

    f = lambda x: -laplace_u(x)
    hat_u = linear_solvers.solve_lu(pr, lower, upper, pc, rhs.rhs(d, n, f))
    print("max. abs. error for d=3, n=10:", rhs.compute_error(d, n, hat_u, u))
    print()

    print("Plot of max. abs. error of Poisson problem approximation for d=3 with respect to N:")
    print()
    rhs.plot_error(d, u, laplace_u, n_start=2, n_end=20)

    print("Plot of condition of matrix A(3) with respect to N:")
    print()
    block_matrix.plot_condition(d, n_start=2, n_end=15)

    print("Plot of number of non-zero entries of Matrix A(3), and its LU decomposition with respect"
          + " to N:")
    block_matrix.plot_non_zero(d, n_start=2, n_end=15)


if __name__ == "__main__":
    main()
