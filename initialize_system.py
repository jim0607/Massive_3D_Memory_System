from math import asinh, atan, sqrt, pi, sin, atan2
import numpy as np
from constant import system_size, mesh


# define random unit vectors
def gen_rand_vecs(dims, number):
    """
    :param dims: dimension of the vector
    :type number: number of vectors
    """
    vecs = np.random.normal(size = (number, dims))

    # np.linalg.norm returns the Norm of the matrix or vector(s).
    mags = np.linalg.norm(vecs, axis = -1)

    return vecs / mags[..., np.newaxis]


# generate random unit vectors in 3D in amount of n1*n2*n3
rand_vec = gen_rand_vecs(3, np.prod(system_size)).reshape(system_size + (3,))

# abs of R(x,y) with random modules
r_xy = np.random.standard_normal(np.prod(system_size)).reshape(system_size)

# initialize magnetization - m is a 4D list
# the first dimension is n_x on the grid;
# the second dimension is n_y on the grid;
# the third dimension is n_z on the grid;
# the fourth dimension refers to (m_x, m_y, m_x])
mangetization = np.zeros(system_size + (3,))
mangetization[1:-1, :, :, 0] = 1.0
mangetization[(-1, 0), :, :, 1] = 1.0
# print(len(mangetization), len(mangetization[0]), len(mangetization[0][0]), len(mangetization[0][0][0]))
# print(mangetization)
