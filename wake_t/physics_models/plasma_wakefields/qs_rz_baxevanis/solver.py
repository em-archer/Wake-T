"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
from numba import njit
import scipy.interpolate as scint
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.charge_deposition import charge_distribution_cyl
from wake_t.particles.susceptibility_deposition import deposit_susceptibility_cyl


# For debugging
# from time import time
# np.seterr(all='raise')
# import matplotlib
# matplotlib.use('Qt5agg')
# import matplotlib.pyplot as plt


def calculate_wakefields(laser, beam_part, r_max, xi_min, xi_max, n_r, n_xi,
                         ppc, n_p, laser_z_foc, p_shape='linear'):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters:
    -----------
    laser : LaserPulse (optional)
        Laser driver of the plasma stage.

    beam_part : list
        List of numpy arrays containing the spatial coordinates and charge of
        all beam particles, i.e [x, y, xi, q].

    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.

    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.

    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.

    n_r : int
        Number of grid elements along r in which to calculate the wakefields.

    n_xi : int
        Number of grid elements along xi in which to calculate the wakefields.

    ppc : int (optional)
        Number of plasma particles per 1d cell along the radial direction.

    n_p : float
        Plasma density in units of m^{-3}.

    laser_z_foc : float
        Focal position of the laser along z in meters. It is measured as
        the distance from the beginning of the PlasmaStage. A negative
        value implies that the focal point is located before the
        PlasmaStage.

    p_shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.

    """
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d

    # Laser parameters.
    if laser is not None:
        laser_params = [laser.a_0, laser.l_0, laser.w_0, laser.tau, laser.xi_c,
                        laser.polarization, laser_z_foc]
    else:
        laser_params = None

    # Initialize plasma particles.
    dr = r_max / n_r
    dr_p = dr / ppc
    n_part = n_r * ppc
    r = np.linspace(dr_p / 2, r_max - dr_p / 2, n_part)
    pr = np.zeros_like(r)
    gamma = np.ones_like(r)
    q = dr_p * r

    # Iteration steps.
    dxi = (xi_max - xi_min) / n_xi

    # Initialize field arrays.
    psi_mesh = np.zeros((n_r, n_xi))
    dr_psi_mesh = np.zeros((n_r, n_xi))
    dxi_psi_mesh = np.zeros((n_r, n_xi))
    b_theta_bar_mesh = np.zeros((n_r, n_xi))
    b_theta_0_mesh = np.zeros((n_r, n_xi))
    r_arr = np.linspace(dr / 2, r_max - dr / 2, n_r)
    xi_arr = np.linspace(xi_min, xi_max, n_xi)

    # Calculate beam source term (b_theta_0) from particle distribution.
    if beam_part is not None:
        beam_source = get_beam_function(
            beam_part, n_r, n_xi, n_p, r_arr, xi_arr, p_shape)
    else:
        beam_source = None

# use line 92 for radius, initialize using (r,0,xi_max)

    # Calculate the charge distribution of the initial column
    rho = np.zeros((n_xi + 4, n_r + 4))
    rho = charge_distribution_cyl(np.full_like(r, xi_max), r_arr, np.zeros_like(r), q, xi_arr[0], r[0], n_xi, n_part, dxi, dr_p, rho, p_shape=p_shape)

    # Calculate the plasma susceptibility of the initial column:
    chi = np.zeros((n_xi + 4, n_r + 4))
    chi = deposit_susceptibility_cyl(np.full_like(r, xi_max), r_arr, np.zeros_like(r), q, xi_arr[0], r[0], n_part, dxi, dr_p, chi, np.ones_like(r), p_shape=p_shape)

    # Main loop.
    for step in np.arange(n_xi):
        xi = xi_max - dxi * step

        # Evolve plasma to next xi step.
        r, pr = evolve_plasma(
            r, pr, q, xi, dxi, laser_params, beam_source, s_d)

        # Remove plasma particles leaving simulation boundaries (plus margin).
        idx_keep = np.where(r <= r_max + 0.1)
        r = r[idx_keep]
        pr = pr[idx_keep]
        gamma = gamma[idx_keep]
        q = q[idx_keep]

        if r.shape[0] == 0:
            break
        # Calculate fields at specified r locations.
        fields = calculate_fields(r_arr, xi, r, pr, q,
                                  laser_params, beam_source, s_d)

        # Deposit charge of updated plasma column using (r, 0, xi)
        rho = charge_distribution_cyl(np.full_like(r, xi), r, np.zeros_like(r), q, xi_arr[0], r[0], n_xi, n_part, dxi, dr_p, rho, p_shape=p_shape)

        # Deposit chi of updated plasma column using (r,0,xi):
        chi = deposit_susceptibility_cyl(np.full_like(r, xi), r, np.zeros_like(r), q, xi_arr[0], r[0], n_part, dxi, dr_p, chi, np.ones_like(r), p_shape=p_shape)

        i = -1 - step

        # Unpack fields.
        (psi_mesh[:, i], dr_psi_mesh[:, i], dxi_psi_mesh[:, i],
         b_theta_bar_mesh[:, i], b_theta_0_mesh[:, i]) = fields

    # Calculate derived fields (E_r, n_p, K_r and E_z').
    dr_psi_mesh, dxi_psi_mesh = np.gradient(psi_mesh, dr, dxi, edge_order=2)
    dxi_psi_mesh *= -1
    dr_psi_mesh *= -1
    e_r_mesh = b_theta_bar_mesh + b_theta_0_mesh - dr_psi_mesh
    r_arr_v = np.vstack(r_arr)
    n_p = (np.gradient(r_arr_v * e_r_mesh, dr, axis=0, edge_order=2) / r_arr_v
           - np.gradient(dxi_psi_mesh, dxi, axis=1, edge_order=2) - 1)
    k_r_mesh = np.gradient(dr_psi_mesh, dr, axis=0, edge_order=2)
    e_z_p_mesh = np.gradient(dxi_psi_mesh, dxi, axis=1, edge_order=2)
    return (n_p, dr_psi_mesh, dxi_psi_mesh, e_z_p_mesh, k_r_mesh, psi_mesh,
            xi_arr, r_arr)


def evolve_plasma(r, pr, q, xi, dxi, laser_params, beam_source, s_d):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using a Runge-Kutta method of 4th order.

    This means that the transverse coordinates are updated as:
    r += (Ar + 2*Br + 2*Cr) + Dr) / 6
    pr += (Apr + 2*Bpr + 2*Cpr) + Dpr) / 6

    The required constants are calculated here and then passed to
    the jittable method 'update_particles_rk4' to apply the equations above.

    Parameters:
    -----------

    r : ndarray
        Array containing the radial position of the particles.

    pr : ndarray
        Array containing the radial momentum of the particles.

    q : ndarray
        Array containing the charge of the particles.

    xi : float
        Current xi position (speed-of-light frame) of the plasma particles.

    dxi : float
        Longitudinal step for the Runge-Kutta solver.

    laser_params : list
        List containing the relevant parameters of the laser pulse,if
        present. Otherwise this parameter is None.

    beam_source : function
        Interpolator function for the azimuthal magnetic field from the
        beam particle distribution, if present. Otherwise this parameter
        is None.

    s_d : float
        Skin depth of the plasma.

    """
    Ar, Apr = motion_derivatives(
        dxi, xi, r, pr, q, laser_params, beam_source, s_d)
    Br, Bpr = motion_derivatives(
        dxi, xi - dxi / 2, r + Ar / 2, pr + Apr / 2, q, laser_params,
        beam_source, s_d)
    Cr, Cpr = motion_derivatives(
        dxi, xi - dxi / 2, r + Br / 2, pr + Bpr / 2, q, laser_params,
        beam_source, s_d)
    Dr, Dpr = motion_derivatives(
        dxi, xi - dxi, r + Cr, pr + Cpr, q, laser_params, beam_source, s_d)
    return update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr)


def motion_derivatives(dxi, xi, r, pr, q, laser_params, beam_source, s_d):
    """
    Return the derivatives of the radial position and momentum of the plasma
    particles.

    The method corrects for any particles with r < 0, calculates the source
    terms for the derivatives and delegates their calculation to the jittable
    method 'calculate_derivatives'.

    For details about the input parameters, check 'evolve_plasma' method.

    """
    # Check for particles with negative radial position. If so, invert them.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        # Make copy to avoid altering data for next Runge-Kutta step.
        r = r.copy()
        pr = pr.copy()
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.

    # Convert xi and r from normalized to SI units.
    xi_si = xi * s_d
    r_si = r * s_d

    # Calculate source terms from laser and beam particles.
    if laser_params is not None:
        nabla_a = get_nabla_a(xi_si, r_si, *laser_params) * s_d
        a2 = get_a2(xi_si, r_si, *laser_params)
    else:
        nabla_a = np.zeros(r.shape)
        a2 = np.zeros(r.shape)
    b_theta_0 = beam_source(r, xi)

    # Calculate motion derivatives in jittable method.
    return calculate_derivatives(dxi, r, pr, q, b_theta_0, nabla_a, a2)


@njit()
def calculate_derivatives(dxi, r, pr, q, b_theta_0, nabla_a, a2):
    """
    Jittable method to which the calculation of the motion derivatives is
    outsourced.

    Parameters:
    -----------
    dxi : float
        Longitudinal step for the Runge-Kutta solver.

    r : ndarray
        Array containing the radial position of the particles.

    pr : ndarray
        Array containing the radial momentum of the particles.

    q : ndarray
        Array containing the charge of the particles.

    b_theta_0 : ndarray
        Array containing the value of the azimuthal magnetic field from
        the beam distribution at the position of each particle.

    nabla_a : ndarray
        Array containing the value of the gradient of the laser normalized
        vector potential at the position of each particle.

    a2 : ndarray
        Array containing the value of the square of the laser normalized
        vector potential at the position of each particle.

    """
    # Preallocate arrays.
    n_part = r.shape[0]
    dr = np.empty(n_part)
    dpr = np.empty(n_part)
    gamma = np.empty(n_part)

    # Calculate wakefield potential and its derivaties at particle positions.
    psi, dr_psi, dxi_psi = calculate_psi_and_derivatives_at_particles(r, pr, q)

    # Calculate gamma (Lorentz factor) of particles.
    for i in range(n_part):
        psi_i = psi[i]
        gamma[i] = (1. + pr[i] ** 2 + a2[i] + (1. + psi_i) ** 2) / (
                2. * (1. + psi_i))

    # Calculate azimuthal magnetic field from plasma at particle positions.
    b_theta_bar = calculate_b_theta_at_particles(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a)

    # Calculate derivatives of r and pr.
    for i in range(n_part):
        psi_i = psi[i]
        dpr[i] = dxi * (gamma[i] * dr_psi[i] / (1. + psi_i)
                        - b_theta_bar[i]
                        - b_theta_0[i]
                        - nabla_a[i] / (2. * (1. + psi_i)))
        dr[i] = dxi * pr[i] / (1. + psi_i)
    return dr, dpr


@njit()
def update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr):
    """
    Jittable method to which updating the particle coordinates in the RK4
    algorithm is outsourced.

    It also checks and corrects for any particles with r < 0.

    """
    # Push particles
    inv_6 = 1. / 6.
    for i in range(r.shape[0]):
        r[i] += (Ar[i] + 2. * (Br[i] + Cr[i]) + Dr[i]) * inv_6
        pr[i] += (Apr[i] + 2. * (Bpr[i] + Cpr[i]) + Dpr[i]) * inv_6
    # Check if any have a negative radial position. If so, invert them.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.
    return r, pr


def calculate_fields(r_arr, xi, r, pr, q, laser_params, beam_source, s_d):
    """
    Calculates the wakefield potential and its derivatives, as well as the
    azimuthal magnetic field from the plasma and beam particles at the
    specified radial locations.

    Parameters:
    -----------
    r_arr : ndarray
        1D array containing the radial positions at which to evaluate the
        fields. This array should be sorted.

    xi : float
        Longitudinal position (speed-of-light frame) at which to evaluate the
        fields. It should also correspond to the current longitudinal
        position of the plasma particles.

    r : ndarray
        Array containing the radial position of the plasma particles.

    pr : ndarray
        Array containing the radial momentum of the plasma particles.

    q : ndarray
        Array containing the charge of the plasma particles.

    laser_params : list
        List containing the relevant parameters of the laser pulse,if
        present. Otherwise this parameter is None.

    beam_source : function
        Interpolator function for the azimuthal magnetic field from the
        beam particle distribution, if present. Otherwise this parameter
        is None.

    s_d : float
        Skin depth of the plasma.

    """
    # Convert xi and r from normalized to SI units.
    xi_si = xi * s_d
    r_si = r * s_d

    # Calculate source terms from laser and beam at plasma particles.
    if laser_params is not None:
        nabla_a = get_nabla_a(xi_si, r_si, *laser_params) * s_d
        a2 = get_a2(xi_si, r_si, *laser_params)
    else:
        nabla_a = np.zeros(r.shape)
        a2 = np.zeros(r.shape)
    b_theta_0 = beam_source(r, xi)

    # Calculate wakefield potential and derivatives at plasma particles.
    psi, dr_psi, dxi_psi = calculate_psi_and_derivatives_at_particles(r, pr, q)
    gamma = (1 + pr ** 2 + a2 / 2 + (1 + psi) ** 2) / (2 * (1 + psi))

    # Calculate all fields at the specified r_arr locations.
    b_theta_0_r = beam_source(r_arr, xi)
    psi_r, dr_psi_r, dxi_psi_r = calculate_psi_and_derivatives(r_arr, r, pr, q)
    b_theta_bar_r = calculate_b_theta(
        r_arr, r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a)
    return psi_r, dr_psi_r, dxi_psi_r, b_theta_bar_r, b_theta_0_r


@njit()
def calculate_psi_and_derivatives_at_particles(r, pr, q):
    """
    Calculate the wakefield potential and its derivatives at the position
    of the plasma particles. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    For details about the input parameters see method 'calculate_fields'.

    """
    # Initialize arrays.
    n_part = r.shape[0]
    psi = np.zeros(n_part)
    dr_psi = np.zeros(n_part)
    dxi_psi = np.zeros(n_part)

    # Initialize value of sums.
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate psi and dr_psi.
    idx = np.argsort(r)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]

        # Calculate new sums.
        sum_1_new = sum_1 + q_i
        sum_2_new = sum_2 + q_i * np.log(r_i)

        # Calculate average.
        sum_1_avg = 0.5 * (sum_1 + sum_1_new)
        sum_2_avg = 0.5 * (sum_2 + sum_2_new)

        # Calculate psi and dr_psi.
        psi[i] = sum_1_avg * np.log(r_i) - sum_2_avg - 0.25 * r_i ** 2
        dr_psi[i] = sum_1_avg / r_i - 0.5 * r_i

        # Update value of sums.
        sum_1 = sum_1_new
        sum_2 = sum_2_new
    r_N = r[-1]
    psi = psi - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)

    # Calculate dxi_psi.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi[i]

        sum_3_new = sum_3 + (q_i * pr_i) / (r_i * (1 + psi_i))
        dxi_psi[i] = -0.5 * (sum_3 + sum_3_new)
        sum_3 = sum_3_new
    dxi_psi = dxi_psi + sum_3
    return psi, dr_psi, dxi_psi


@njit()
def calculate_psi_and_derivatives(r_arr, r, pr, q):
    """
    Calculate the wakefield potential and its derivatives at the radial
    positions specified in r_arr. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    For details about the input parameters see method 'calculate_fields'.

    """
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]
    psi_part = np.zeros(n_part)
    sum_1_arr = np.zeros(n_part)
    sum_2_arr = np.zeros(n_part)
    sum_3_arr = np.zeros(n_part)
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    idx = np.argsort(r)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
        psi_part[i] = sum_1 * np.log(r_i) - sum_2 - 0.25 * r_i ** 2
    r_N = r[-1]
    psi_part += - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)

    # Calculate sum_3.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi_part[i]

        sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
        sum_3_arr[i] = sum_3

    # Initialize arrays for psi and derivatives at r_arr locations.
    n_points = r_arr.shape[0]
    psi = np.zeros(n_points)
    dr_psi = np.zeros(n_points)
    dxi_psi = np.zeros(n_points)

    # Calculate fields at r_arr.
    i_last = 0
    for j in range(n_points):
        r_j = r_arr[j]
        # Get index of last plasma particle with r_i < r_j.
        for i_sort in range(n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            psi[j] = -0.25 * r_j ** 2
            dr_psi[j] = -0.5 * r_j
            dxi_psi[j] = 0.
        else:
            i_p = idx[i_last]
            psi[j] = sum_1_arr[i_p] * np.log(r_j) - sum_2_arr[
                i_p] - 0.25 * r_j ** 2
            dr_psi[j] = sum_1_arr[i_p] / r_j - 0.5 * r_j
            dxi_psi[j] = - sum_3_arr[i_p]
    psi = psi - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)
    dxi_psi = dxi_psi + sum_3
    return psi, dr_psi, dxi_psi


@njit()
def calculate_b_theta_at_particles(r, pr, q, gamma, psi, dr_psi, dxi_psi,
                                   b_theta_0, nabla_a):
    """
    Calculate the azimuthal magnetic field from the plasma at the location
    of the plasma particles using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters:
    -----------
    r_arr : ndarray
        1D array containing the radial positions at which to evaluate the
        fields. This array should be sorted.

    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.

    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.

    b_theta_0, nabla_a : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0, idx = calculate_ai_bi(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a)

    # Calculate field at particles as average between neighboring values.
    n_part = r.shape[0]
    a_im1 = a_0
    b_im1 = 0.
    a_i_avg = np.zeros(n_part)
    b_i_avg = np.zeros(n_part)
    for i_sort in range(n_part):
        i = idx[i_sort]
        a_i_avg[i] = 0.5 * (a_i[i] + a_im1)
        b_i_avg[i] = 0.5 * (b_i[i] + b_im1)
        a_im1 = a_i[i]
        b_im1 = b_i[i]
    b_theta_bar = a_i_avg * r + b_i_avg / r
    return b_theta_bar


@njit()
def calculate_b_theta(r_arr, r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                      nabla_a):
    """
    Calculate the azimuthal magnetic field from the plasma at the radial
    locations in r_arr using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    Parameters:
    -----------
    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.

    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.

    b_theta_0, nabla_a : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0, idx = calculate_ai_bi(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a)

    # Calculate fields at r_arr
    n_part = r.shape[0]
    n_points = r_arr.shape[0]
    b_theta_mesh = np.zeros(n_points)
    i_last = 0
    for j in range(n_points):
        r_j = r_arr[j]
        # Get index of last plasma particle with r_i < r_j.
        for i_sort in range(n_part):
            i_p = idx[i_sort]
            r_i = r[i_p]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta_mesh[j] = a_0 * r_j
        else:
            i_p = idx[i_last]
            b_theta_mesh[j] = a_i[i_p] * r_j + b_i[i_p] / r_j

    return b_theta_mesh


@njit()
def calculate_ai_bi(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters see method 'calculate_b_theta'.

    The values of a_i and b_i are calculated as follows, using Eqs. (26) and
    (27) from the paper of P. Baxevanis and G. Stupakov:

        Write a_i and b_i as linear system of a_0:

            a_i = K_i * a_0 + T_i
            b_i = U_i * a_0 + P_i


        Where (im1 stands for subindex i-1):

            K_i = (1 + A_i*r_i/2) * K_im1  +  A_i/(2*r_i)     * U_im1
            U_i = (-A_i*r_i**3/2) * K_im1  +  (1 - A_i*r_i/2) * U_im1

            T_i = ( (1 + A_i*r_i/2) * T_im1  +  A_i/(2*r_i)     * P_im1  +
                    (2*Bi + Ai*Ci)/4 )
            P_i = ( (-A_i*r_i**3/2) * T_im1  +  (1 - A_i*r_i/2) * P_im1  +
                    r_i*(4*Ci - 2*Bi*r_i - Ai*Ci*r_i)/4 )

        With initial conditions:

            K_0 = 1
            U_0 = 0
            O_0 = 0
            P_0 = 0

        Then a_0 can be determined by imposing a_N = 0:

            a_N = K_N * a_0 + O_N = 0 <=> a_0 = - O_N / K_N

    """
    n_part = r.shape[0]

    # Preallocate arrays
    K = np.zeros(n_part)
    U = np.zeros(n_part)
    T = np.zeros(n_part)
    P = np.zeros(n_part)

    # Establish initial conditions (K_0 = 1, U_0 = 0, O_0 = 0, P_0 = 0)
    K_im1 = 1.
    U_im1 = 0.
    T_im1 = 0.
    P_im1 = 0.

    # Iterate over particles
    idx = np.argsort(r)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        gamma_i = gamma[i]
        psi_i = psi[i]
        dr_psi_i = dr_psi[i]
        dxi_psi_i = dxi_psi[i]
        b_theta_0_i = b_theta_0[i]
        nabla_a_i = nabla_a[i]

        a = 1. + psi_i
        a2 = a * a
        a3 = a2 * a
        b = 1. / (r_i * a)
        c = 1. / (r_i * a2)
        pr_i2 = pr_i * pr_i

        A_i = q_i * b
        B_i = q_i * (- (gamma_i * dr_psi_i) * c
                     + (pr_i2 * dr_psi_i) / (r_i * a3)
                     + (pr_i * dxi_psi_i) * c
                     + pr_i2 / (r_i * r_i * a2)
                     + b_theta_0_i * b
                     + nabla_a_i * c * 0.5)
        C_i = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)

        l_i = (1. + 0.5 * A_i * r_i)
        m_i = 0.5 * A_i / r_i
        n_i = -0.5 * A_i * r_i ** 3
        o_i = (1. - 0.5 * A_i * r_i)

        K_i = l_i * K_im1 + m_i * U_im1
        U_i = n_i * K_im1 + o_i * U_im1
        T_i = l_i * T_im1 + m_i * P_im1 + 0.5 * B_i + 0.25 * A_i * C_i
        P_i = n_i * T_im1 + o_i * P_im1 + r_i * (
                C_i - 0.5 * B_i * r_i - 0.25 * A_i * C_i * r_i)

        K[i] = K_i
        U[i] = U_i
        T[i] = T_i
        P[i] = P_i

        K_im1 = K_i
        U_im1 = U_i
        T_im1 = T_i
        P_im1 = P_i

    # Calculate a_0.
    a_0 = - T_im1 / K_im1

    # Calculate a_i and b_i as functions of a_0.
    a_i = K * a_0 + T
    b_i = U * a_0 + P
    return a_i, b_i, a_0, idx


def get_beam_function(beam_part, n_r, n_xi, n_p, r_arr, xi_arr, p_shape):
    """
    Return a function of r and xi which gives the azimuthal magnetic field
    from a particle distribution. This is Eq. (18) in the original paper.

    For details about input parameters see method 'calculate_wakefields'.

    """
    # Plasma skin depth.
    s_d = ge.plasma_skin_depth(n_p / 1e6)

    # Grid parameters.
    dr = r_arr[1] - r_arr[0]
    dxi = xi_arr[1] - xi_arr[0]
    r_min = r_arr[0]
    xi_min = xi_arr[0]

    # Grid arrays with guard cells.
    r_grid_g = (0.5 + np.arange(-2, n_r + 2)) * dr
    xi_grid_g = np.arange(-2, n_xi + 2) * dxi + xi_min

    # Get and normalize particle coordinate arrays.
    x, y, xi, q = beam_part
    xi_n = xi / s_d
    x_n = x / s_d
    y_n = y / s_d

    # Calculate particle weights.
    w = q / ct.e / (2 * np.pi * dr * dxi * s_d ** 3 * n_p)

    q_dist = np.zeros((n_xi + 4, n_r + 4))
    # Obtain charge distribution (using cubic particle shape by default).
    q_dist = charge_distribution_cyl(
        xi_n, x_n, y_n, w, xi_min, r_min, n_xi, n_r, dxi, dr, q_dist, p_shape=p_shape)

    # Calculate radial integral (Eq. (18)).
    r_int = np.cumsum(q_dist, axis=1) / np.abs(r_grid_g) * dr

    # Create and return interpolator.
    return scint.interp2d(r_grid_g, xi_grid_g, -r_int)


@njit()
def get_nabla_a(xi, r, a_0, l_0, w_0, tau, xi_c, pol='linear', dz_foc=0):
    """ Calculate the gradient of the normalized vector potential. """
    z_r = np.pi * w_0 ** 2 / l_0
    w_fac = np.sqrt(1 + (dz_foc / z_r) ** 2)
    s_r = w_0 * w_fac / np.sqrt(2)
    s_z = tau * ct.c / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2)
    avg_amplitude = a_0
    if pol == 'linear':
        avg_amplitude /= np.sqrt(2)
    return - 2 * (avg_amplitude / w_fac) ** 2 * r / s_r ** 2 * (
                np.exp(-r ** 2 / (s_r ** 2)) * np.exp(
            -(xi - xi_c) ** 2 / (s_z ** 2)))


@njit()
def get_a2(xi, r, a_0, l_0, w_0, tau, xi_c, pol='linear', dz_foc=0):
    """ Calculate the square of the normalized vector potential. """
    z_r = np.pi * w_0 ** 2 / l_0
    w_fac = np.sqrt(1 + (dz_foc / z_r) ** 2)
    s_r = w_0 * w_fac / np.sqrt(2)
    s_z = tau * ct.c / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2)
    avg_amplitude = a_0
    if pol == 'linear':
        avg_amplitude /= np.sqrt(2)
    return (avg_amplitude / w_fac) ** 2 * (np.exp(-(r) ** 2 / (s_r ** 2)) *
                                           np.exp(
                                               -(xi - xi_c) ** 2 / (s_z ** 2)))
