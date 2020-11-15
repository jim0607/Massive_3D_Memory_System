from math import asinh, atan, sqrt, pi, sin, atan2

""""
Define the constants used in the simulator
"""

"""
1. Parameters for the 3D system
"""

system_size = (10, 10, 1)  # 3D system size 100 x 100 x 1
mesh = (100e-9, 100e-9, 30e9)  # define the mesh of the 3D system
delta_t = 1e-13  # unit in second
length = system_size[0] * mesh[0]  # m
width = system_size[1] * mesh[1]  # m


"""
2. Parameters for the material
"""

saturation_magnetization = 8e5  # unit in A/m
Mpd3fe = 5.0e5  # A/m
exchange_coeff = 1e-11  # J/m
dmi_strength = 1e-10    # J/m
constant_fiend = Mpd3fe * 5e-3  # A/m
damping_constant = 0.5
anisotropy_coeff_ku_1 = 1e6    # anisotropy constant
anisotropy_coeff_ku_2 = 0      # anisotropy constant


"""
3. Physics Constant in equations
"""

eps = 1e-9
mu0 = 1.2566e-6  # N/exchange_coeff^2
Flux_0 = 2.0678e-15  # Wb
gamma = 2.211e5
oe2am = 79.5775  # multiply it with Oe to convert to exchange_coeff/m
