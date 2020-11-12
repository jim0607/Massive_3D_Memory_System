import numpy as np
from matplotlib import pyplot as plt
import os

curr_folder = os.getcwd()
curr_folder = os.path.join(curr_folder, "data")

data = np.loadtxt('data/standard_problem.dat')
data1 = np.loadtxt("data/standard_problem_test.dat")

t = data[:, 0]
mx = data[:, 1]
my = data[:, 2]
mz = data[:, 3]

t1 = data1[:, 0]
mx1 = data1[:, 1]
my1 = data1[:, 2]
mz1 = data1[:, 3]

plt.subplot(3, 1, 1)
plt.plot(t, mx)
plt.plot(t1, mx1, '-')
plt.title('Test for a standard problem')
plt.ylabel('<mx>')

plt.subplot(3, 1, 2)
plt.plot(t, my)
plt.plot(t1, my1, '-')
plt.ylabel('<my>')

plt.subplot(3, 1, 3)
plt.plot(t, -mz)
plt.plot(t1, mz1, '-')
plt.ylabel('<mz>')
plt.xlabel('time (ns)')

plt.tight_layout()

save_fig_name = "Test_for_a_standard_problem" + ".png"
plt.savefig(os.path.join(curr_folder, save_fig_name), bbox_inches = "tight", dpi = 1200)

plt.show()