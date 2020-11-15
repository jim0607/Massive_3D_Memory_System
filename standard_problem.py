from constant import mu0, system_size
from initialize_system import mangetization
import numpy as np
from solver import llg_rk4_evolver
from plot_system_state import plot_magnetization

import os


# gradient_descent
alpha = 0.5
for i in range(5000):
    llg_rk4_evolver(mangetization, 1e-13)
print "Relaxed"
plot_magnetization(mangetization)

# switch
alpha = 0.02
dt = 1e-12

# numpy.tile construct an array by repeating exchange_coeff the number of times given by reps
# numpy.prod return the product of array elements over a given axis
h_zee = np.tile([-24.6e-3 / mu0, +4.3e-3 / mu0, 0.0], np.prod(system_size)).reshape(mangetization.shape)


curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")
with open(curr_folder + '/standard_problem.dat', 'w') as f:
    for i in range(int(10e-10 / dt)):
        f.write("%f %f %f %f\n" % ((i * 1e9 * dt,) + tuple(map(lambda i: np.mean(mangetization[:, :, :, i]), range(3)))))
        m0 = np.copy(mangetization)
        mangetization = llg_rk4_evolver(mangetization, dt, h_zee)           # run llg_rk4 is pretty much like making one prediction

plot_magnetization(mangetization)
