from math import asinh, atan, sqrt, pi, sin, atan2

"""
Parameters for the materials system
"""
Ms = 8e5  # unit in A/m
n = (10, 10, 1)  # 3D system size 100 x 100 x 1
Mpd3fe = 5.0e5  # A/m
A = 1e-11  # J/m
DMI_strength = 1e-10    # J/m
Hs = Mpd3fe * 5e-3  # A/m
alpha = 0.5
Ku_1 = 1e6    # anisotropy constant
Ku_2 = 0      # anisotropy constant

dm = 230e-9  # m
df = 30e-9  # m
dx = (100e-9, 100e-9, df)  # m*m*m
dt = 1e-13  # s
a = n[0] * dx[0]  # m
b = n[1] * dx[1]  # m
eps = 1e-9
mu0 = 1.2566e-6  # N/A^2
Flux_0 = 2.0678e-15  # Wb
gamma = 2.211e5
oe2am = 79.5775  # multiply it with Oe to convert to A/m
