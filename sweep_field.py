import numpy as np
from constant import oe2am, n, dt
from initialize_system import m
from plot_system_state import plot_m, save_m
from solver import relax
from get_magnetization import m_x, m_y, m_z, m_mean
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")


H = np.arange(-500, 501, 100) * oe2am  # in A/m
data = []
i = 0
for k in H:
    save_m(m, save_fig_name = "time =" + str(i) + "s")
    h_zee = np.tile([k, 0.0, 0.0], np.prod(n)).reshape(m.shape)
    print "H = %e A/m" % (k)
    relax(m, dt, h_zee)
    mx = m_x(m)
    my = m_y(m)
    mz = m_z(m)
    data.append([mx, my, mz, k])
    i += 1

np.savetxt(curr_folder + "\Hx_sweep.dat", data)