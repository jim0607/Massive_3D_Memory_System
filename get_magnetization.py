from constant import system_size
import numpy as np


class GetMagnetization:
    """
    get the magnetization in x, y and z direction from the 3D system
    """

    def __init__(self, magnetization):
        """
        @type magnetization: object
        """
        self.magnetization = magnetization


    def mean_x(self):
        """
        compute the mean of magnetization in the x direction

        @type magnetization: object
        """
        return np.mean(self.magnetization[:, :, :, 0])


    def mean_y(self):
        """
        compute the mean of magnetization in the y direction

        @type magnetization: object
        """
        return np.mean(self.magnetization[:, :, :, 1])


    def mean_z(self):
        """
        compute the mean of magnetization in the z direction

        @type magnetization: object
        """
        return np.mean(self.magnetization[:, :, :, 2])


    def mean_magnetization(self):
        """
        compute the mean of magnetization in the x, y and z direction

        @type magnetization: object
        """
        m_1 = self.mean_x(self.magnetization)
        m_2 = self.mean_y(self.magnetization)
        m_3 = self.mean_z(self.magnetization)
        mean_m = np.sqrt(m_1 * m_1 + m_2 * m_2 + m_3 * m_3)

        return mean_m