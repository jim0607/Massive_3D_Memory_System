import numpy as np


def mean_x(magnetization):
    """
    compute the mean of magnetization in the x direction

    @type magnetization: object
    """
    return np.mean(magnetization[:, :, :, 0])


# <m_y>
def mean_y(magnetization):
    """
    compute the mean of magnetization in the y direction

    @type magnetization: object
    """
    return np.mean(magnetization[:, :, :, 1])


# <m_z>
def mean_z(magnetization):
    """
    compute the mean of magnetization in the z direction

    @type magnetization: object
    """
    return np.mean(magnetization[:, :, :, 2])


def mean_magnetization(magnetization):
    """
    compute the mean of magnetization in the x, y and z direction

    @type magnetization: object
    """
    m_1 = mean_x(magnetization)
    m_2 = mean_y(magnetization)
    m_3 = mean_z(magnetization)
    mean_m = np.sqrt(m_1 * m_1 + m_2 * m_2 + m_3 * m_3)

    return mean_m