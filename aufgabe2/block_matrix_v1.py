"""
    Date: 15.11.2019
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np



class BlockMatrix:

    def __init__(self, d, n):
        constructor_list = [0 for i in range(n-3)] + [-1, 2*d, -1] + [0 for i in range(n-3)]
        matrix = np.zeros((n-1, n-1))

        for z_index in range(len(matrix)):
            print(constructor_list[n-z_index-2 : 2*n-z_index-3])
            matrix[z_index] = np.array(constructor_list[n-z_index-2 : 2*n-z_index-3])

        l = 1
        print("l =", l)
        print(matrix)

        l = 1
        while l in range(d):
            l += 1

            m_new = np.zeros(((n-1)**l, (n-1)**l))
            b_size = (n-1)**(l-1)

            for z_index in range(len(m_new)):

                z_list = [0 for i in range(int(z_index/b_size - 1) * b_size)]

                if int(z_index/b_size) > 0:
                    z_list += [0 for i in range(z_index % b_size)] + [-1]
                    z_list += [0 for i in range(b_size - z_index % b_size - 1)]

                z_list += list(matrix[z_index % b_size])

                if int(z_index/b_size) < 2:
                    z_list += [0 for i in range(z_index % b_size)] + [-1]
                    z_list += [0 for i in range(b_size - z_index % b_size - 1)]

                z_list += [0 for i in range((n-3 - int(z_index/b_size)) * b_size)]

                print(z_list)
                m_new[z_index] = np.array(z_list)

            matrix = m_new
            print("l =", l)
            print(matrix)

        self.matrix = matrix


    def get_sparse(self):
        pass

    def eval_zeros(self):
        n_zeros = 0
        for z_index in range(len(self.matrix)):
            for s_index in range(len(self.matrix)):
                if self.matrix[z_index, s_index] == 0:
                    n_zeros += 1

        total_cells = len(self.matrix)**2

        return total_cells - n_zeros, n_zeros, 1 - n_zeros/total_cells, n_zeros/total_cells




def main():
    m = BlockMatrix(3, 4)
    print()
    print()
    print(m.eval_zeros())


if __name__ == "__main__":
    main()


