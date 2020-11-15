import numpy as np
from constant import oe2am, system_size, delta_t
from initialize_system import mangetization
from plot_system_state import plot_magnetization, save_magnetization_fig
from solver import gradient_descent
from get_magnetization import mean_x, mean_y, mean_z, mean_magnetization
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")


H = np.arange(-500, 501, 100) * oe2am  # in exchange_coeff/m
data = []
i = 0
for k in H:
    save_magnetization_fig(mangetization, save_fig_name = "time =" + str(i) + "s")
    h_zee = np.tile([k, 0.0, 0.0], np.prod(system_size)).reshape(mangetization.shape)
    print "H = %e A/m" % (k)
    gradient_descent(mangetization, delta_t, h_zee)
    mx = mean_x(mangetization)
    my = mean_y(mangetization)
    mz = mean_z(mangetization)
    data.append([mx, my, mz, k])
    i += 1

np.savetxt(curr_folder + "\Hx_sweep.dat", data)