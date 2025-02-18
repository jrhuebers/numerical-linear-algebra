"""
    Date: 15.11.2019
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python

from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack

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
