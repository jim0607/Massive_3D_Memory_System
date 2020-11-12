import numpy as np

# <m_x>
def m_x(m):
    return np.mean(m[:, :, :, 0])


# <m_y>
def m_y(m):
    return np.mean(m[:, :, :, 1])


# <m_z>
def m_z(m):
    return np.mean(m[:, :, :, 2])


def m_mean(m):
    m_1 = m_x(m)
    m_2 = m_y(m)
    m_3 = m_z(m)
    m_mean = np.sqrt(m_1 * m_1 + m_2 * m_2 + m_3 * m_3)
    return m_mean