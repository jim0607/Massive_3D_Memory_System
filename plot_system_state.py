from matplotlib import pyplot as plt
from constant import n
import numpy as np
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")


def plot_m(m, k=1):
    X, Y = np.mgrid[0:n[0], 0:n[1]]
    U = m[:, :, :, 0].flatten()
    V = m[:, :, :, 1].flatten()

    # plt.axis([0.025, 0.025, 0.95, 0.95])

    plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
               units = 'dots', color = 'Teal', headlength = 7)

    plt.xlim(-1, n[0])
    plt.xticks(())
    plt.ylim(-1, n[1])
    plt.yticks(())

    plt.show()


def save_m(m, k=1, save_fig_name = ""):
    X, Y = np.mgrid[0:n[0], 0:n[1]]
    U = m[:, :, :, 0].flatten()
    V = m[:, :, :, 1].flatten()

    # plt.axis([0.025, 0.025, 0.95, 0.95])

    plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
               units = 'dots', color = 'Teal', headlength = 5)

    plt.xlim(-1, n[0])
    plt.xticks(())
    plt.ylim(-1, n[1])
    plt.yticks(())

    plt.savefig(os.path.join(curr_folder, save_fig_name), bbox_inches = "tight", dpi = 240)
