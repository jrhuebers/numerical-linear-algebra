"""
    Date: 15.11.2019
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np


class BlockMatrix:

    def __init__(self, d, n):
        l = 1

        matrix = 2*d*np.eye(n-1) - np.eye(n-1, k=-1) - np.eye(n-1, k=1)

        while l in range(d):
            l += 1
            block_size = (n-1)**(l-1)

            matrices = [np.zeros((block_size, block_size)) for i in range(n-3)]
            neg_diag_matrix = np.zeros(block_size) - np.eye(block_size)
            matrices += [neg_diag_matrix, matrix, neg_diag_matrix]
            matrices += [np.zeros((block_size, block_size)) for i in range(n-3)]


            matrix_layers = [np.concatenate(matrices[n-i-2:2*n-i-3], axis=1)
                             for i in range(n-1)]

            matrix = np.concatenate(matrix_layers, axis=0)

        self.matrix = matrix


    def get_sparse(self):
        val_list = []
        coord_list = []
        for row_ind in range(len(self.matrix)):
            for col_ind in range(len(self.matrix)):
                if self.matrix[row_ind, col_ind] != 0:
                    val_list.append(self.matrix[row_ind, col_ind])
                    coord_list.append((row_ind, col_ind))

        return val_list, coord_list


    def eval_zeros(self):
        n_zeros = 0
        for row_ind in range(len(self.matrix)):
            for col_ind in range(len(self.matrix)):
                if self.matrix[row_ind, col_ind] == 0:
                    n_zeros += 1

        total_cells = len(self.matrix)**2

        return total_cells - n_zeros, n_zeros, 1 - n_zeros/total_cells, n_zeros/total_cells




def main():
    obj = BlockMatrix(3, 3)
    print("BlockMatrix(3, 3):")
    print(obj.matrix)
    print()

    print("Anzahl von nicht-Nullen / Nullen / Anteil von nicht-Nullen / von Nullen:")
    print(obj.eval_zeros())
    print()

    print("BlockMatrix(3, 3) im sparse-Format:")
    tup = obj.get_sparse()
    print(tup[0])
    print(tup[1])


if __name__ == "__main__":
    main()
