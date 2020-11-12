import numpy as np
from constant import n


def save_m(m, f):
    np.savetxt(f, (m[:, :, :, 0].flatten(), m[:, :, :, 1].flatten(), m[:, :, :, 2].flatten()))


def load_m(f):
    k = np.loadtxt(f)
    j = np.zeros(n + (3,))
    for i in range(3): j[:, :, :, i] = k[i, :].reshape(j[:, :, :, i].shape)
    return j


def load_H_sweep(f):
    n = np.loadtxt(f)
    return n[:, 0:3], n[:, 3]