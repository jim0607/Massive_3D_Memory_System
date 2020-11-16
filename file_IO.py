import numpy as np
from constant import system_size


class FileIO:
    """
    file IO read the data and write the data
    """

    def save_magnetization(magnetization, file):
        """
        save magnetization to file

        @type magnetization: object
        """
        np.savetxt(file, (magnetization[:, :, :, 0].flatten(),
                          magnetization[:, :, :, 1].flatten(),
                          magnetization[:, :, :, 2].flatten()))


    def load_magnetization(file):
        """
        load magnetization from the imput file

        @type file: object
        """
        mag_file = np.loadtxt(file)
        magnetization = np.zeros(system_size + (3,))

        for i in range(3):
            magnetization[:, :, :, i] = mag_file[i, :].reshape(magnetization[:, :, :, i].shape)

        return magnetization


    def load_field_sweep(file):
        """
        load field distribution from input file

        @type file: object
        """
        n = np.loadtxt(file)

        return n[:, 0:3], n[:, 3]