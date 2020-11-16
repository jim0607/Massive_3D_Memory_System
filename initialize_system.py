from math import asinh, atan, sqrt, pi, sin, atan2
import numpy as np
from constant import system_size, mesh


class InitializeSystem:
    """
    we assume the initial magnetization in the 3D system is a random state
    we initialize the 3D system by setting the the magnetization randomly distributed
    """

    def __init__(self, dims, number):
        """
        @type dims: the dims of the random vector
        @type number: the number of random vectors
        """
        self.dims = dims
        self.number = number


    def gen_rand_vecs(self):
        """
        define random unit vectors

        @return: the random unit vector for the scheduled 3D system
        """
        vecs = np.random.normal(size = (self.number, self.dims))

        # np.linalg.norm returns the Norm of the matrix or vector(s).
        mags = np.linalg.norm(vecs, axis = -1)

        return vecs / mags[..., np.newaxis]


    def get_random_vecs(self):
        """
        generate the random vectors

        @return: the random vectors
        """
        rand_vec = self.gen_rand_vecs().reshape(system_size + (self.dims,))

        # abs of R(x,y) with random modules
        r_xy = np.random.standard_normal(np.prod(system_size)).reshape(system_size)

        return rand_vec, r_xy


    def get_init_magnetization(self):
        """
        generate random unit vectors in 3D in amount of n1*n2*n3

        @return: the randomly distributed magnetization
        """
        # initialize magnetization - m is a 4D list
        # the first dimension is n_x on the grid;
        # the second dimension is n_y on the grid;
        # the third dimension is n_z on the grid;
        # the fourth dimension refers to (m_x, m_y, m_x])
        magnetization = np.zeros(system_size + (self.dims,))
        magnetization[1:-1, :, :, 0] = 1.0
        magnetization[(-1, 0), :, :, 1] = 1.0
        # print(len(magnetization), len(magnetization[0]), len(magnetization[0][0]), len(magnetization[0][0][0]))
        # print(magnetization)

        return magnetization
