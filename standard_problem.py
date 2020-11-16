from constant import mu0, system_size, damping_constant
from initialize_system import InitializeSystem
import numpy as np
from minimizers_gradient_descent import Minimizers
from plottings import Plottings
import os

"""
Below is a demo for a standard problem: 
switching the magnetization by an external magnetization
"""

init_sys = InitializeSystem(3, np.prod(system_size))
magnetization = init_sys.get_init_magnetization()
plottings = Plottings(magnetization)
minimizers = Minimizers()

for i in range(5000):
    minimizers.llg_rk4_evolver(magnetization, 1e-13)
print "Relaxed"
plottings.plot_magnetization(k=1)


alpha = 0.02
dt = 1e-12

# numpy.tile construct an array by repeating exchange_coeff the number of times given by reps
# numpy.prod return the product of array elements over a given axis
h_zee = np.tile([-24.6e-3 / mu0, +4.3e-3 / mu0, 0.0], np.prod(system_size)).reshape(magnetization.shape)

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")

with open(curr_folder + '/standard_problem.dat', 'w') as f:
    for i in range(int(10e-10 / dt)):
        f.write("%f %f %f %f\n" % ((i * 1e9 * dt,) + tuple(map(lambda i: np.mean(magnetization[:, :, :, i]), range(3)))))
        m0 = np.copy(magnetization)
        magnetization = minimizers.llg_rk4_evolver(magnetization, dt, h_zee)           # run llg_rk4 is pretty much like making one prediction

plottings.plot_magnetization(k=1)
