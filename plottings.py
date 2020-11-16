from matplotlib import pyplot as plt
from constant import system_size
import numpy as np
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")

class Plottings:
    """
    we need to plot system states, magnetization versus space and magnetization versus time
    """

    def __init__(self, magnetization):
        self.magnetization = magnetization


    def plot_magnetization(self, k=1):
        """
        plot magnetization

        @type magnetization: object
        """
        X, Y = np.mgrid[0:system_size[0], 0:system_size[1]]
        U = self.magnetization[:, :, :, 0].flatten()
        V = self.magnetization[:, :, :, 1].flatten()

        # plt.axis([0.025, 0.025, 0.95, 0.95])

        plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
                   units = 'dots', color = 'Teal', headlength = 7)

        plt.xlim(-1, system_size[0])
        plt.xticks(())
        plt.ylim(-1, system_size[1])
        plt.yticks(())

        plt.show()


    def save_magnetization_fig(self, k=1, save_fig_name =""):
        """
        plot magnetization

        @type magnetization: object
        """
        X, Y = np.mgrid[0:system_size[0], 0:system_size[1]]
        U = self.magnetization[:, :, :, 0].flatten()
        V = self.magnetization[:, :, :, 1].flatten()

        # plt.axis([0.025, 0.025, 0.95, 0.95])

        plt.quiver(X[::k, ::k], Y[::k, ::k], U[::k], V[::k],
                   units = 'dots', color = 'Teal', headlength = 5)

        plt.xlim(-1, system_size[0])
        plt.xticks(())
        plt.ylim(-1, system_size[1])
        plt.yticks(())

        plt.savefig(os.path.join(curr_folder, save_fig_name), bbox_inches = "tight", dpi = 240)


    def plot_magnetization_vs_time(self):
        """
        plot the magnetization versus time
        """
        data = np.loadtxt('data/standard_problem.dat')
        data1 = np.loadtxt("data/standard_problem_test.dat")

        time = data[:, 0]
        mx = data[:, 1]
        my = data[:, 2]
        mz = data[:, 3]

        time1 = data1[:, 0]
        mx1 = data1[:, 1]
        my1 = data1[:, 2]
        mz1 = data1[:, 3]

        plt.subplot(3, 1, 1)
        plt.plot(time, mx)
        plt.plot(time1, mx1, '-')
        plt.title('Test for a standard problem')
        plt.ylabel('<mx>')

        plt.subplot(3, 1, 2)
        plt.plot(time, my)
        plt.plot(time1, my1, '-')
        plt.ylabel('<my>')

        plt.subplot(3, 1, 3)
        plt.plot(time, -mz)
        plt.plot(time1, mz1, '-')
        plt.ylabel('<mz>')
        plt.xlabel('time (ns)')

        plt.tight_layout()

        save_fig_name = "Test_for_a_standard_problem" + ".png"
        plt.savefig(os.path.join(curr_folder, save_fig_name), bbox_inches = "tight", dpi = 1200)

        plt.show()
