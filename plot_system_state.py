from matplotlib import pyplot as plt
from constant import system_size
import numpy as np
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")


def plot_magnetization(magnetization, k=1):
    """
    plot magnetization

    @type magnetization: object
    """
    X, Y = np.mgrid[0:system_size[0], 0:system_size[1]]
    U = magnetization[:, :, :, 0].flatten()
    V = magnetization[:, :, :, 1].flatten()

    # plt.axis([0.025, 0.025, 0.95, 0.95])

    plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
               units = 'dots', color = 'Teal', headlength = 7)

    plt.xlim(-1, system_size[0])
    plt.xticks(())
    plt.ylim(-1, system_size[1])
    plt.yticks(())

    plt.show()


def save_magnetization_fig(magnetization, k=1, save_fig_name =""):
    """
    plot magnetization

    @type magnetization: object
    """
    X, Y = np.mgrid[0:system_size[0], 0:system_size[1]]
    U = magnetization[:, :, :, 0].flatten()
    V = magnetization[:, :, :, 1].flatten()

    # plt.axis([0.025, 0.025, 0.95, 0.95])

    plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
               units = 'dots', color = 'Teal', headlength = 5)

    plt.xlim(-1, system_size[0])
    plt.xticks(())
    plt.ylim(-1, system_size[1])
    plt.yticks(())

    plt.savefig(os.path.join(curr_folder, save_fig_name), bbox_inches = "tight", dpi = 240)
