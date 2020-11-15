import numpy as np
from demag_tensor import f_n_demag, m_pad
from constant import system_size, mesh, saturation_magnetization, exchange_coeff, mu0, constant_fiend, Mpd3fe, anisotropy_coeff_ku_1, anisotropy_coeff_ku_2
from initialize_system import rand_vec, r_xy


def initialize_magnetization(magnetization):
    """
    Initialize magnetization field

    @type magnetization: object
    @return initialized magnetization in the 3D system
    """

    e_dot_m = (rand_vec * magnetization).sum(axis = 3)  # scalar product of e*m
    R_xy = e_dot_m * r_xy  # modules of R(x,y)*m(x,y)

    initialized_mag_field = rand_vec * R_xy.reshape(system_size + (1,))  # anisotropy random vectors

    return initialized_mag_field


def calculate_demagnetization_field(magnetization):
    """
    calculate demagnetization field using Fourier Transform, refer to the two papers below
    W. Scholz, J. Fidler, T. Schre, D. Suess, R. Dittrich, H. Forster, and V. Tsiantos,
    "Scal-able parallel micromagnetic solvers for magnetic nanostructures,"
    Computational Materials Science, vol. 28, no. 2, pp. 366-383, 2003.

    D. Suess, V. Tsiantos, T. Schre, J. Fidler, W. Scholz, H. Forster, R. Dittrich, and J. Miles,
    "Time resolved micromagnetics using a preconditioned time integration method"
    Journal of Magnetism and Magnetic Materials, vol. 248, no. 2, pp. 298-311, 2002.

    @type magnetization: object
    @return demagnetization field
    """

    m_pad[:system_size[0], :system_size[1], :system_size[2], :] = magnetization
    f_m_pad = np.fft.rfftn(m_pad, axes = filter(lambda i: system_size[i] > 1, range(3)))

    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype = f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis = 3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis = 3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis = 3)
    h_demag = np.fft.irfftn(f_h_demag_pad, axes = filter(lambda i: system_size[i] > 1, range(3)))[:system_size[0], :system_size[1], :system_size[2], :]

    return h_demag


def calculate_exchange_field(magnetization):
    """
    calculate the exchange field, using the 6-neighbors exchange coupling method
    Krawczyk, Maciej, et al.
    "On the formulation of the exchange field in the Landau-Lifshitz equation for spin-wave calculation in magnonic crystals."
    Advances in Condensed Matter Physics 2012 (2012).

    @type magnetization: object
    @return exchange field
    """

    h_ex = - 2 * magnetization * sum([1 / x ** 2 for x in mesh])
    for i in range(6):  # we adopt six-neighbor model for exchange coupling calculation
        if system_size[i % 3] == 1:
            h_ex += np.repeat(magnetization, 1, axis = i % 3) / mesh[i % 3] ** 2
        else:
            h_ex += np.repeat(magnetization, [i / 3 * 2] + [1] * (system_size[i % 3] - 2) + [2 - i / 3 * 2], axis = i % 3) / mesh[i % 3] ** 2

    return h_ex


def uniaxial_anisotropy(magnetization):
    """
    calculate anisotropy field. here we only care about the 1st fold anisotropy field
    Ding, Jinjun, et al. "Nanometer-thick yttrium iron garnet films with perpendicular anisotropy and low damping."
    Physical Review Applied 14.1 (2020): 014017.

    @type magnetization: object
    """
    h_anisotropy = 2 * anisotropy_coeff_ku_1 / saturation_magnetization

    return h_anisotropy


def uniaxial_anisotropy_2(magnetization):
    """
    calculate anisotropy field. here we only care about both the 1st fold anisotropy field, and the 2nd fold anisotropy field
    Ding, Jinjun, et al. "Nanometer-thick yttrium iron garnet films with perpendicular anisotropy and low damping."
    Physical Review Applied 14.1 (2020): 014017.

    @type magnetization: object
    """
    h_anisotropy_1 = 2 * anisotropy_coeff_ku_1 / saturation_magnetization
    h_anisotropy_2 = 2 * anisotropy_coeff_ku_2 / saturation_magnetization

    return h_anisotropy_1 + h_anisotropy_2


def calculate_effective_field(magnetization):
    """
    compute the effective field, which include initialized magnetization, demagnetization field, exchange field and anisotropy field

    @type magnetization: object
    """
    initialized_field = initialize_magnetization(magnetization) * constant_fiend * (saturation_magnetization / Mpd3fe)
    demag_field = (calculate_demagnetization_field(magnetization) * saturation_magnetization) * (saturation_magnetization / Mpd3fe)
    exchange_field = 2 * exchange_coeff / (mu0 * saturation_magnetization) * calculate_exchange_field(magnetization) * (saturation_magnetization / Mpd3fe)
    anisotropy_field = uniaxial_anisotropy_2(magnetization) * (saturation_magnetization / Mpd3fe)

    return initialized_field + demag_field + exchange_field + anisotropy_field
