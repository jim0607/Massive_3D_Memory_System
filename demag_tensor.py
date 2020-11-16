from math import asinh, atan, sqrt, pi, sin, atan2
import numpy as np
from constant import system_size, mesh


class AuxiliaryFunctions:
    """
    auxiliary function to calculate demag tensor
    """

    def __init__(self, eps=1e-18):
        # a very small number to avoid divide by zero
        self.eps = eps

        # setup demag tensor
        self.n_demag = np.zeros([1 if i == 1 else 2 * i for i in system_size] + [6])
        for i, t in enumerate(((self.aux_f, 0, 1, 2), (self.aux_g, 0, 1, 2), (self.aux_g, 0, 2, 1),
                               (self.aux_f, 1, 2, 0), (self.aux_g, 1, 2, 0), (self.aux_f, 2, 0, 1))):
            self.set_n_demag(i, t[1:], t[0])

        self.m_pad = np.zeros([1 if i == 1 else 2 * i for i in system_size] + [3])

        self.f_n_demag = np.fft.rfftn(self.n_demag, axes = filter(lambda i: system_size[i] > 1, range(3)))


    def aux_f(self, point):
        """
        auxiliary function to calculate demag tensor:
        J. Newell, W. Williams, and D. J. Dunlop "exchange_coeff generalization of the demagnetizing tensor for nonuniform magnetization"

        @type p: object
        """
        x, y, z = abs(point[0]), abs(point[1]), abs(point[2])

        return + y / 2.0 * (z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + self.eps)) \
               + z / 2.0 * (y ** 2 - x ** 2) * asinh(z / (sqrt(x ** 2 + y ** 2) + self.eps)) \
               - x * y * z * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + self.eps)) \
               + 1.0 / 6.0 * (2 * x ** 2 - y ** 2 - z ** 2) * sqrt(x ** 2 + y ** 2 + z ** 2)


    def aux_g(self, point):
        """
        auxiliary function to calculate demag tensor:
        J. Newell, W. Williams, and D. J. Dunlop "exchange_coeff generalization of the demagnetizing tensor for nonuniform magnetization"

        @type point: object
        """
        x, y, z = abs(point[0]), abs(point[1]), abs(point[2])

        return + x * y * z * asinh(z / (sqrt(x ** 2 + y ** 2) + self.eps)) \
               + y / 6.0 * (3.0 * z ** 2 - y ** 2) * asinh(x / (sqrt(y ** 2 + z ** 2) + self.eps)) \
               + x / 6.0 * (3.0 * z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + self.eps)) \
               - z ** 3 / 6.0 * atan(x * y / (z * sqrt(x ** 2 + y ** 2 + z ** 2) + self.eps)) \
               - z * y ** 2 / 2.0 * atan(x * z / (y * sqrt(x ** 2 + y ** 2 + z ** 2) + self.eps)) \
               - z * x ** 2 / 2.0 * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + self.eps)) \
               - x * y * sqrt(x ** 2 + y ** 2 + z ** 2) / 3.0


    def set_n_demag(self, dimension, permute, func):
        """
        demag tensor setup

        @type permute: object
        """

        # numpy.nditer is an efficient multi-dimensional iterator object to iterate over arrays
        it = np.nditer(self.n_demag[:, :, :, dimension], flags = ['multi_index'], op_flags = ['writeonly'])

        while not it.finished:
            value = 0.0

            for i in np.rollaxis(np.indices((2,) * 6), 0, 7).reshape(64, 6):
                idx = map(lambda k: (it.multi_index[k] + system_size[k]) % (2 * system_size[k]) - system_size[k], range(3))
                value += (-1) ** sum(i) * func(map(lambda j: (idx[j] + i[j] - i[j + 3]) * mesh[j], permute))

            it[0] = - value / (4 * pi * np.prod(mesh))
            it.iternext()
