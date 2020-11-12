from constant import gamma, alpha, pi, atan2, err
import numpy as np
from field_calculation import h_eff


# compute llg right side
def llg_rhs(m, h_eff, h_zee):
    h = h_eff(m) + h_zee
    llg_right_hand_side = - gamma / (1 + alpha ** 2) * np.cross(m, h) - alpha * gamma / (1 + alpha ** 2) * np.cross(m, np.cross(m, h))
    return llg_right_hand_side


# compute llg step using Euler method
def llg_eu(m, dt, h_zee=0.0):
    dm_dt = llg_rhs(m, h_eff, h_zee)
    print ("error = %g" % (dm_dt * dt).mean())
    m += dt * dm_dt
    return m / np.repeat(np.sqrt((m * m).sum(axis = 3)), 3).reshape(m.shape)


# compute llg step using RK4
def llg_rk4(m, dt, h_zee=0.0):
    k1 = llg_rhs(m, h_eff, h_zee)
    k2 = llg_rhs(m + (dt / 2.0) * k1, h_eff, h_zee)
    k3 = llg_rhs(m + (dt / 2.0) * k2, h_eff, h_zee)
    k4 = llg_rhs(m + dt * k3, h_eff, h_zee)
    dm_dt = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    m += dt * dm_dt
    m = m / np.repeat(np.sqrt((m * m).sum(axis = 3)), 3).reshape(m.shape)
    return m


# relaxing
def relax(m, dt, h_zee=0.0):
    m0 = np.copy(m)
    llg_rk4(m, dt, h_zee)
    deg = ((180 / pi) * atan2(np.sqrt(((m - m0) ** 2).sum(3)).max(), np.sqrt(((m0) ** 2).sum(3)).max()))
    dm = np.sqrt(((m - m0) ** 2).sum(3)).max()

    while dm > err:
        m0 = np.copy(m)
        llg_rk4(m, dt, h_zee)
        deg = ((180 / pi) * atan2(np.sqrt(((m - m0) ** 2).sum(3)).max(), np.sqrt(((m0) ** 2).sum(3)).max()))
        dm = np.sqrt(((m - m0) ** 2).sum(3)).max()

    print "deg = %e, dm = %e" % (deg, dm)
