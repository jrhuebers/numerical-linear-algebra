"""
    Module block_matrix.py
    Date: 15.11.2019

    Implements a class BlockMatrix representing sparse matrices and various utility functions
    including a method to calculate the LU-decomposition of a matrix.
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python


from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
import scipy.sparse.linalg as sla

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension

    Attributes
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension
    """

    def __init__(self, d, n):
        self.d = d
        self.n = n


    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            block_matrix in a csr sparse data format
        """

        d = self.d
        n = self.n

        matrix = lil_matrix((n-1, n-1))
        matrix.setdiag(2*d)
        if n > 2:
            matrix.setdiag(-1, k=1)
            matrix.setdiag(-1, k=-1)

        matrix = matrix.tocsr()

        for l in range(2, d+1):
            block = (n-1)**(l-1)

            matrices = [csc_matrix((block, block)) for i in range(n-3)]
            neg_diag_matrix = lil_matrix((block, block))
            neg_diag_matrix.setdiag(-1)
            neg_diag_matrix = neg_diag_matrix.tocsc()
            matrices += [neg_diag_matrix, matrix.tocsc(), neg_diag_matrix]
            matrices += [csc_matrix((block, block)) for i in range(n-3)]

            matrix_layers = [hstack(matrices[n-i-2:2*n-i-3]).tocsr() for i in range(n-1)]

            matrix = vstack(matrix_layers)

        return matrix


    def eval_zeros(self):
        """ Returns the (absolute and relative) numbers of (non-)zero elements
        of the matrix. The relative number of the (non-)zero elements are with
        respect to the total number of elements of the matrix.

        Returns
        -------
        int
            number of non-zeros
        int
            number of zeros
        float
            relative number of non-zeros
        float
            relative number of zeros
        """

        n_non_zero = 1
        for l in range(1, self.d+1):
            n_non_zero *= (self.n-1)
            n_non_zero += 2*(self.n-2)*((self.n-1)**(l-1))

        total_cells = (self.n-1)**(2*self.d)

        return n_non_zero, total_cells-n_non_zero, n_non_zero/total_cells, 1-n_non_zero/total_cells

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form pr * A * pc = l * u

        Returns
        -------
        pr : scipy.sparse.csr_matrix
            row permutation matrix of LU-decomposition
        l : scipy.sparse.csr_matrix
            lower triangular unit diagonal matrix of LU-decomposition
        u : scipy.sparse.csr_matrix
            upper triangular matrix of LU-decomposition
        pc : scipy.sparse.csr_matrix
            column permutation matrix of LU-decomposition
        """

        lu = sla.splu(self.get_sparse())

        n = lu.shape[0]
        Pr = csc_matrix((np.ones(n), (lu.perm_r, np.arange(n))))
        Pc = csc_matrix((np.ones(n), (np.arange(n), lu.perm_c)))

        return Pr, lu.L, lu.U, Pc

    def eval_zeros_lu(self):
        """ Returns the absolute and relative numbers of (non-)zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        int
            Number of zeros
        float
            Relative number of non-zeros
        float
            Relative number of zeros
        """

        L, U = self.get_lu()[1:3]

        total_cells = (self.n-1)**(2*self.d)

        non_zero = 0
        for num in L.data:
            if num != 0:
                non_zero += 1
        for num in U.data:
            if num != 0:
                non_zero += 1

        non_zero -= (self.n-1)**self.d

        return non_zero, total_cells-non_zero, non_zero/total_cells, 1-non_zero/total_cells

    def get_cond(self):
        """ Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to max-norm
        """

        matrix = self.get_sparse()
        norm_a = (abs(matrix).dot(np.ones((self.n-1)**self.d))).max()
        norm_inv = (abs(sla.inv(matrix)).dot(np.ones((self.n-1)**self.d))).max()
        return norm_a*norm_inv


def plot_condition(d, n_start=2, n_end=10):
    """ Plots the conditions of matrix A(d) with respect to N.
    """

    matrix = BlockMatrix(d, 1)
    n_values = range(n_start, n_end+1)

    conditions = []
    for n in n_values:
        matrix.n = n
        conditions.append(matrix.get_cond())

    plt.plot(n_values, conditions)
    plt.title("Condition of A(d) with respect to N")
    plt.xlabel("N")
    plt.ylabel("Condition")
    plt.show()

def plot_non_zero(d, n_start=2, n_end=10):
    """ Plots the number of non-zero entries in matrix A(d) and its LU-decomposition
    with respect to N.
    """

    matrix = BlockMatrix(d, 1)
    n_values = range(n_start, n_end+1)

    non_zero_a = []
    non_zero_lu = []
    for n in n_values:
        matrix.n = n
        tmp = matrix.eval_zeros()
        non_zero_a.append(tmp[0])
        tmp = matrix.eval_zeros_lu()
        non_zero_lu.append(tmp[0])

    plt.plot(n_values, non_zero_a, label="A(d)")
    plt.plot(n_values, non_zero_lu, label="LU-decomposition")
    plt.title("Non-zeros in A(d) and LU-decomposition with respect to N")
    plt.xlabel("N")
    plt.ylabel("Non-zero entries")
    plt.yscale("log")
    plt.legend()
    plt.show()


def main():
    """ main-Function of block_matrix.py, executed when pragram started as main program.
    Creates an instance of class BlockMatrix of size (3, 3) and demonstrates methods get_sparse()
    and eval_zeros() by printing their output. The Matrix is also printed in dense form for
    verification of the results.
    """

    obj = BlockMatrix(3, 3)

    print("BlockMatrix(3, 3) in sparse form:")
    print(obj.get_sparse())
    print()

    print("Number of non-zeros / zeros / proportion of non-zeros to total entries / of zeros:")
    print(obj.eval_zeros())
    print()

    print("BlockMatrix(3, 3) in dense form:")
    print(obj.get_sparse().toarray())


if __name__ == "__main__":
    main()
