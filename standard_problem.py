from constant import mu0, n
from initialize_system import m
import numpy as np
from solver import llg_rk4
from plot_system_state import plot_m

import os


# relax
alpha = 0.5
for i in range(5000):
    llg_rk4(m, 1e-13)
print "Relaxed"
plot_m(m)

# switch
alpha = 0.02
dt = 1e-12

# numpy.tile construct an array by repeating A the number of times given by reps
# numpy.prod return the product of array elements over a given axis
h_zee = np.tile([-24.6e-3 / mu0, +4.3e-3 / mu0, 0.0], np.prod(n)).reshape(m.shape)


curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")
with open(curr_folder + '/standard_problem.dat', 'w') as f:
    for i in range(int(10e-10 / dt)):
        f.write("%f %f %f %f\n" % ((i * 1e9 * dt,) + tuple(map(lambda i: np.mean(m[:, :, :, i]), range(3)))))
        m0 = np.copy(m)
        m = llg_rk4(m, dt, h_zee)           # run llg_rk4 is pretty much like making one prediction

plot_m(m)
