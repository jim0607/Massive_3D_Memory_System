from math import asinh, atan, sqrt, pi, sin, atan2
import numpy as np
from constant import n, dx

# a very small number
eps = 1e-18


# auxiliary function to calculate demag tensor:
# A. J. Newell, W. Williams, and D. J. Dunlop "A generalization of the demagnetizing tensor for nonuniform magnetization"
def f(p):
    x, y, z = abs(p[0]), abs(p[1]), abs(p[2])
    return + y / 2.0 * (z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + eps))\
           + z / 2.0 * (y ** 2 - x ** 2) * asinh(z / (sqrt(x ** 2 + y ** 2) + eps))\
           - x * y * z * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + eps))\
           + 1.0 / 6.0 * (2 * x ** 2 - y ** 2 - z ** 2) * sqrt(x ** 2 + y ** 2 + z ** 2)


# auxiliary function to calculate demag tensor:
# A. J. Newell, W. Williams, and D. J. Dunlop "A generalization of the demagnetizing tensor for nonuniform magnetization"
def g(p):
    x, y, z = abs(p[0]), abs(p[1]), abs(p[2])
    return + x * y * z * asinh(z / (sqrt(x ** 2 + y ** 2) + eps))\
           + y / 6.0 * (3.0 * z ** 2 - y ** 2) * asinh(x / (sqrt(y ** 2 + z ** 2) + eps))\
           + x / 6.0 * (3.0 * z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + eps))\
           - z ** 3 / 6.0 * atan(x * y / (z * sqrt(x ** 2 + y ** 2 + z ** 2) + eps))\
           - z * y ** 2 / 2.0 * atan(x * z / (y * sqrt(x ** 2 + y ** 2 + z ** 2) + eps))\
           - z * x ** 2 / 2.0 * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + eps))\
           - x * y * sqrt(x ** 2 + y ** 2 + z ** 2) / 3.0


# demag tensor setup
def set_n_demag(c, permute, func):
    # numpy.nditer is an efficient multi-dimensional iterator object to iterate over arrays
    it = np.nditer(n_demag[:, :, :, c], flags = ['multi_index'], op_flags = ['writeonly'])
    while not it.finished:
        value = 0.0
        for i in np.rollaxis(np.indices((2,) * 6), 0, 7).reshape(64, 6):
            idx = map(lambda k: (it.multi_index[k] + n[k]) % (2 * n[k]) - n[k], range(3))
            value += (-1) ** sum(i) * func(map(lambda j: (idx[j] + i[j] - i[j + 3]) * dx[j], permute))
        it[0] = - value / (4 * pi * np.prod(dx))
        it.iternext( )


# setup demag tensor
n_demag = np.zeros([1 if i == 1 else 2 * i for i in n] + [6])
for i, t in enumerate(((f, 0, 1, 2), (g, 0, 1, 2), (g, 0, 2, 1), (f, 1, 2, 0), (g, 1, 2, 0), (f, 2, 0, 1))):
    set_n_demag(i, t[1:], t[0])

m_pad = np.zeros([1 if i == 1 else 2 * i for i in n] + [3])
f_n_demag = np.fft.rfftn(n_demag, axes = filter(lambda i: n[i] > 1, range(3)))
