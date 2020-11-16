import numpy as np
from constant import oe2am, system_size, delta_t
from initialize_system import InitializeSystem
from plottings import Plottings
from minimizers_gradient_descent import Minimizers
from get_magnetization import GetMagnetization
import os


"""
Below is a demo for a standard problem: 
switching the magnetization by sweeping the external magnetization
"""

H = np.arange(-500, 501, 100) * oe2am  # in exchange_coeff/m

init_sys = InitializeSystem(3, np.prod(system_size))
magnetization = init_sys.get_init_magnetization()
plottings = Plottings(magnetization)

data = []
i = 0

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")

for k in H:
    plottings.save_magnetization_fig(save_fig_name = "time =" + str(i) + "s")

    h_zee = np.tile([k, 0.0, 0.0], np.prod(system_size)).reshape(magnetization.shape)
    print "H = %e A/m" % (k)

    minimizers = Minimizers()
    minimizers.gradient_descent(magnetization, delta_t, h_zee)

    get_mag = GetMagnetization(magnetization)
    mx = get_mag.mean_x()
    my = get_mag.mean_y()
    mz = get_mag.mean_z()

    data.append([mx, my, mz, k])

    i += 1

np.savetxt(curr_folder + "\Hx_sweep.dat", data)