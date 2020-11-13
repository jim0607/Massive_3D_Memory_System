from math import asinh, atan, sqrt, pi, sin, atan2
import numpy as np
from constant import n, dx


# define random unit vectors
def gen_rand_vecs(dims, number):
    """
    :param dims: dimension of the vector
    :type number: object
    """
    vecs = np.random.normal(size = (number, dims))
    mags = np.linalg.norm(vecs, axis = -1)
    return vecs / mags[..., np.newaxis]


e = gen_rand_vecs(3, np.prod(n)).reshape(n + (3,))  # random unit vectors in 3d in amount of n1*n2*n3
r = np.random.standard_normal(np.prod(n)).reshape(n)  # abs of R(x,y) with random modules


"""
initialize magnetization - m is a 4D list
the first dimension is n_x on the grid;
the second dimension is n_y on the grid;
the third dimension is n_z on the grid;
the fourth dimension refers to (m_x, m_y, m_x])
"""

m = np.zeros(n + (3,))
m[1:-1, :, :, 0] = 1.0
m[(-1, 0), :, :, 1] = 1.0
# print(len(m), len(m[0]), len(m[0][0]), len(m[0][0][0]))
# print(m)
