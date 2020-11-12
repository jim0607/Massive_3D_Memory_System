import numpy as np
from demag_tensor import f_n_demag, m_pad
from constant import n, dx, Ms, A, mu0, Hs, Mpd3fe
from initialize_system import e, r


# demag field
def h_demag(m):
    m_pad[:n[0], :n[1], :n[2], :] = m
    f_m_pad = np.fft.rfftn(m_pad, axes = filter(lambda i: n[i] > 1, range(3)))
    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype = f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis = 3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis = 3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis = 3)
    h_demag = np.fft.irfftn(f_h_demag_pad, axes = filter(lambda i: n[i] > 1, range(3)))[:n[0], :n[1], :n[2], :]
    return h_demag


# exchange field
def h_ex(m):
    h_ex = - 2 * m * sum([1 / x ** 2 for x in dx])
    for i in range(6):      # we adopt six-neighbor model for exchange coupling calculation
        h_ex += np.repeat(m, 1 if n[i % 3] == 1 else [i / 3 * 2] + [1] * (n[i % 3] - 2) + [2 - i / 3 * 2], axis = i % 3) / dx[i % 3] ** 2
    # print 'exchange field computed'
    return h_ex


# IMS
def h_IMS(m):
    e_dot_m = (e * m).sum(axis = 3)  # scalar product of e*m
    R = e_dot_m * r  # modules of R(x,y)*m(x,y)
    h_IMS = e * R.reshape(n + (1,))  # anisotropy random vectors

    return h_IMS


# compute effective field
def h_eff(m):
    return (h_demag(m) * Ms + 2 * A / (mu0 * Ms) * h_ex(m) + h_IMS(m) * Hs) * (Ms / Mpd3fe)
