r"""
Optical Kerr effect for the laser envelope solver.

The envelope solver advances :math:`(\nabla_{tr}^2 + 2i k_0/k_p d/dt + \... ) a = \chi a`, with
all lengths normalized to :math:`1/k_p`. 

The plasma response is included by:
.. math::
            \chi_{plasma} \sim (n/n_p)/gamma.
            
A weak nonlinear dielectric response with relative permittivity change :math`d_{eps}`
can be included in the same way by:
.. math::

    chi_{kerr} = -(k_0/k_p)^2 * d_{eps} ,   d_{eps} = 2 * n_0 * n_2 * I   (Kerr),

so a Kerr medium with :math:`n_2 > 0` gives a *negative* chi that is most negative where
the intensity is highest (focusing), exactly like the relativistic term of a
plasma, :math:`chi_{rel} \approx -(n/n_p)|a|^2/2.
"""

import numpy as np
import scipy.constants as ct


class KerrGas:
    r"""Kerr (:math:`n_2`) response of a neutral gas, for use with a ``LaserPulse``.

    Parameters
    ----------
    n2 : float
        Nonlinear index in m^2/W at the reference density ``n_ref``
        (e.g. 0.8e-23 for H2 at STP).
    n_gas : float
        Gas number density in m^-3 (or any unit, if ``n_ref`` uses the same).
        n2 is scaled linearly with density.
    n_ref : float
        Number density at which :math:`n_2` was quoted. Use the same reference
        (temperature/pressure) as your source for n2.
    n0 : float, optional
        Linear refractive index (default 1). Only enters as the factor :math:`2 n_0`.
    """

    def __init__(self, n2: float, n_gas: float, n_ref: float, n0: float = 1.0):
        self.n2 = n2 * n_gas / n_ref
        self.n0 = n0

    def chi(self, a_env: np.ndarray, k_0: float, k_p: float, polarization="linear"):
        """Return the Kerr susceptibility on the same grid as ``a_env`` (nz x nr)."""
        omega_0 = k_0 * ct.c
        # Intensity per unit |a|^2 (linear polarization, a = peak amplitude).
        I_a = 0.5 * ct.epsilon_0 * ct.c * (ct.m_e * ct.c * omega_0 / ct.e) ** 2
        if polarization == "circular":
            # Same |a| gives twice the intensity, but n2_circ = 2/3 n2_lin.
            I_a, n2 = 2.0 * I_a, self.n2 * 2.0 / 3.0
        else:
            n2 = self.n2
        d_eps = 2.0 * self.n0 * n2 * I_a * np.abs(a_env) ** 2
        return -((k_0 / k_p) ** 2) * d_eps
