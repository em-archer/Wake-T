from typing import Dict, List, Optional, Callable, Union
import math

import numpy as np
import scipy.constants as ct

from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse
from wake_t.utilities.numba import njit_parallel, prange
from wake_t.utilities.other import ProfStart, ProfStop

# Atomic units (SI)
AU_ENERGY = ct.physical_constants["Hartree energy"][0]          # J
AU_ENERGY_EV = AU_ENERGY / ct.e                                  # ~27.211 eV
AU_TIME = ct.hbar / AU_ENERGY                                    # s
AU_EFIELD = AU_ENERGY / (ct.e * ct.physical_constants[
    "Bohr radius"][0])                                           # V/m


@njit_parallel
def do_grid_ionization(
    model,
    num_ion_species,
    elec_density,
    ion_densities,
    chi_array,
    a_env,
    ion_start_index,
    ion_atomic_number,
    ion_mass,
    omega0,
    prefactors,
    is_linear_pol,
    n_xi,
    n_r,
    d_zeta_inv,
):
    dt = 1.0 / (d_zeta_inv * ct.c)

    for i_s in range(num_ion_species):
        ion_density = ion_densities[
            ion_start_index[i_s] : (ion_start_index[i_s] + ion_atomic_number[i_s] + 1),
            2:-1,
            2:-2,
        ]
        chi_factor_ion = ct.m_e / ion_mass[i_s]
        is_last_plasma = i_s + 1 == num_ion_species
        max_ion_lev = ion_atomic_number[i_s]

        for i_r in prange(n_r):
            for i_zeta in range(n_xi - 1, -1, -1):
                Et = 1j * a_env[i_zeta, i_r] * omega0
                if i_zeta + 1 < n_xi:
                    Et += (
                        (a_env[i_zeta + 1, i_r] - a_env[i_zeta, i_r])
                        * ct.c
                        * d_zeta_inv
                    )
                Ep = np.sqrt(np.abs(Et * Et))
                Ep *= ct.m_e * ct.c / ct.e

                chi = 0

                for ion_lev in range(max_ion_lev):
                    p = 0
                    if Ep > 1e-30:
                        if model == "ADK":
                            w_dtau_dc = (
                                prefactors[i_s, ion_lev, 1]
                                * np.pow(Ep, prefactors[i_s, ion_lev, 0])
                                * np.exp(prefactors[i_s, ion_lev, 2] / Ep)
                            )

                            w_dtau_ac = w_dtau_dc
                            if is_linear_pol:
                                w_dtau_ac *= np.sqrt(Ep * prefactors[i_s, ion_lev, 3])

                        elif model == "PPT":
                            gamma = prefactors[i_s, ion_lev, 0] / Ep
                            gamma2 = gamma * gamma
                            sq = np.sqrt(1 + gamma2)
                            beta = 2 * gamma / sq
                            alpha = 2 * np.arcsinh(gamma) - beta

                            kappa = prefactors[i_s, ion_lev, 1]
                            v = kappa * (0.5 + 1.0 / (4 * gamma2))
                            exponent = -2 * v * (
                                np.arcsinh(gamma) - gamma * sq / (1 + 2 * gamma2)
                            )
                            s = (
                                np.sqrt(np.pi) / (2 * (alpha + beta))
                                * np.sqrt(beta / alpha)
                            )
                            
                            w_dtau_dc = (
                                prefactors[i_s, ion_lev, 3]
                                * np.pow(beta, prefactors[i_s, ion_lev, 2])
                                * (gamma2 / (1 + gamma2))
                                * np.exp(exponent)
                                * s
                                * np.sqrt(np.pi * kappa * gamma / 3)
                            )
                            w_dtau_ac = w_dtau_dc
                            if is_linear_pol:
                                w_dtau_ac *= np.sqrt(3 / (np.pi * kappa * gamma))
                            w_dtau_ac *= dt

                        p = 1 - np.exp(-w_dtau_ac)

                    old_weight = (
                        ion_density[ion_lev, i_zeta, i_r]
                        + ion_density[ion_lev, i_zeta + 1, i_r]
                    )
                    transferred_weight = old_weight * p
                    new_weight = old_weight - transferred_weight

                    # ion contribution
                    chi += new_weight * chi_factor_ion * ion_lev * ion_lev

                    ion_density[ion_lev, i_zeta, i_r] = new_weight
                    ion_density[ion_lev + 1, i_zeta, i_r] += transferred_weight
                    elec_density[i_zeta, i_r] += transferred_weight

                ion_density[max_ion_lev, i_zeta, i_r] += ion_density[
                    max_ion_lev, i_zeta + 1, i_r
                ]
                chi += (
                    ion_density[max_ion_lev, i_zeta, i_r]
                    * chi_factor_ion
                    * max_ion_lev
                    * max_ion_lev
                )

                if is_last_plasma:
                    elec_density[i_zeta, i_r] += elec_density[i_zeta + 1, i_r]
                    chi += elec_density[i_zeta, i_r]

                chi_array[i_zeta + 2, i_r + 2] += chi


class LaserGridIonization(RZWakefield):
    """
    This model can be used to propagate laser pulses through a neutral gas or
    plasma that gets ionized from the laser. Specifically, it can be used
    when the laser is too weak to form a plasma wake, namely when a0 << 1.
    This model does not calculate any electric or magnetic fields and should
    not be used with a particle bunch.

    Parameters
    ----------
    density_function : callable
        Function that returns the initial electron density value at the
        given position z. This parameter is given by the `PlasmaStage`
        and does not need to be specified by the user.
        Usually this should be zero.
    ion_species: list
        List of ion species names. Currently supported are H, He, N and Ar.
    ion_densities: list
        List of initial densities for each ion species. Can be a function of z and r.
    r_max : float
        Maximum radial position up to which laser will be calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        laser will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        laser will be calculated.
    n_r : int
        Number of grid elements along r to calculate the laser.
    n_xi : int
        Number of grid elements along xi to calculate the laser.
    model : str
        The ionisation model to use, either ``ADK`` or ``PPT``
    dz_fields : float, optional
        Determines how often the laser and plasma refractive index should
        be updated. If dz_fields=0 (default value), the laser is calculated
        at every step of the Runge-Kutta solver for the beam particle evolution
        (most expensive option). If specified, the laser is only
        updated in steps determined by dz_fields. For example, if
        dz_fields=10e-6, the laser is only updated every time
        the simulation window advances by 10 micron. By default, if not
        specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
        the length the simulation box.
    species_rho_diags : bool, optional
        Whether the model should save the charge density of each plasma species
        separately.
    r_max_plasma : float, optional
        Maximum radial extension of the plasma column. If ``None``, the
        plasma extends up to the ``r_max`` boundary of the simulation box.
    laser : LaserPulse, optional
        Laser driver of the plasma stage.
    laser_evolution : bool, optional
        If True (default), the laser pulse is evolved
        using a laser envelope model. If False, the pulse envelope stays
        unchanged throughout the computation.
    laser_envelope_substeps : int, optional
        Number of substeps of the laser envelope solver per `dz_fields`.
        The time step of the envelope solver is therefore
        `dz_fields / c / laser_envelope_substeps`.
    laser_envelope_nxi, laser_envelope_nr : int, optional
        If given, the laser envelope will run in a grid of size
        (`laser_envelope_nxi`, `laser_envelope_nr`) instead
        of (`n_xi`, `n_r`). This allows the laser to run in a finer (or
        coarser) grid than the plasma refractive index. It is not necessary
        to specify both parameters. If one of them is not given, the resolution
        of the plasma grid will be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.
    field_diags : list, optional
        List of fields to save to openpmd diagnostics.
        One can get the per-ion-level density using 'n_<ion_species>_ionlevel_<level>'
        and the electron density using n_electrons.
        Note that E, B and rho are not set by this model.
        By default ['chi', 'a_mod', 'a_phase', 'a']. Can also be 'all'.
        Each entry can also be a dict to give more precise control of the
        outputted data. The dict can have the following keys:
        {
            "field" : list of field names to output
            "r_min" , "r_max" , "xi_min" , "xi_max" :
                Cut the diagnostic box in r and xi.
            "r_stride" , "xi_stride" :
                Add a stride in units of dr and dxi to r and xi.
            "do_transpose" :
                Wheater to transpose data for the output (default True).
            "diag_name" :
                Name to append to the field name if there are multiple
                diagnostics with the same field.
        }

    See Also
    --------
    Quasistatic2DWakefieldIon

    """

    def __init__(
        self,
        density_function: Callable[[float, float], float],
        ion_species: List[Union[str, Dict]],
        ion_densities: List[Union[float, Callable[[float, float], float]]],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        model: str,
        dz_fields: Optional[float] = None,
        species_rho_diags: Optional[bool] = False,
        r_max_plasma: Optional[float] = None,
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
        field_diags: Optional[List[str]] = None,
    ) -> None:
        if field_diags is None:
            field_diags = ["chi", "a_mod", "a_phase", "a"]
        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            species_rho_diags=species_rho_diags,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            laser_envelope_nxi=laser_envelope_nxi,
            laser_envelope_nr=laser_envelope_nr,
            laser_envelope_use_phase=laser_envelope_use_phase,
            field_diags=field_diags,
            model_name="laser_in_vacuum",
        )
        self.model = model
        assert self.model in ("ADK", "PPT"), (
            f"model must be 'ADK' or 'PPT', got {self.model!r}"
        )
        
        self.r_max_plasma = r_max_plasma

        ion_species_lookup = {
            "H": {"mass_u": 1.007975, "ionization_energy_eV": [13.59843449]},
            "He": {
                "mass_u": 4.002602,
                "ionization_energy_eV": [24.58738880, 54.4177650],
            },
            "N": {
                "mass_u": 14.006855,
                "ionization_energy_eV": [
                    14.53413,
                    29.60125,
                    47.4453,
                    77.4735,
                    97.8901,
                    552.06732,
                    667.046116,
                ],
            },
            "Ar": {
                "mass_u": 39.948,
                "ionization_energy_eV": [
                    15.7596117,
                    27.62967,
                    40.735,
                    59.58,
                    74.84,
                    91.290,
                    124.41,
                    143.4567,
                    422.60,
                    479.76,
                    540.4,
                    619.0,
                    685.5,
                    755.13,
                    855.5,
                    918.375,
                    4120.6656,
                    4426.2228,
                ],
            },
        }

        if not isinstance(ion_species, list):
            ion_species = [ion_species]

        if not isinstance(ion_densities, list):
            ion_densities = [ion_densities]

        self.ion_species = []
        self.ion_names = []

        for species in ion_species:
            if isinstance(species, str):
                if species not in ion_species_lookup:
                    raise ValueError(
                        f"ion_species {species} not found in {list(ion_species_lookup.keys())}"
                    )
                self.ion_species.append(ion_species_lookup[species])
                name = species
            else:
                if (
                    not isinstance(species, dict)
                    or species.keys() != ion_species_lookup["He"].keys()
                ):
                    raise ValueError(
                        f"ion_species {species} must be str or dict like {ion_species_lookup['He']}"
                    )
                self.ion_species.append(species)
                name = "Ion"
            prev_name_cout = self.ion_names.count(name)
            self.ion_names.append(
                name if prev_name_cout == 0 else name + str(prev_name_cout)
            )

        self.initial_ion_densities = []

        for density in ion_densities:
            if isinstance(density, float):

                def uniform_density(z, r, density=density):
                    return np.ones_like(z) * np.ones_like(r) * density

                self.initial_ion_densities.append(uniform_density)
            elif callable(density):
                self.initial_ion_densities.append(density)
            else:
                raise ValueError(
                    f"Type {type(density)} of {density} not supported for ion density."
                )

        self.ion_mass = np.array(
            [species["mass_u"] * ct.u for species in self.ion_species]
        )

        self.ion_atomic_number = np.array(
            [len(species["ionization_energy_eV"]) for species in self.ion_species]
        )

        self.prefactors = np.zeros(
            (len(self.ion_atomic_number), np.max(self.ion_atomic_number), 4)
        )

        if self.model == "PPT":
            self._ppt_prefactors_ready = False

        for i, species in enumerate(self.ion_species):
            wa = (
                ct.alpha**3
                * ct.c
                / ct.physical_constants["classical electron radius"][0]
            )
            Ea = (
                ct.m_e
                * ct.c
                * ct.c
                / ct.e
                * ct.alpha**4
                / ct.physical_constants["classical electron radius"][0]
            )
            UH = ion_species_lookup["H"]["ionization_energy_eV"][0]
            l_eff = np.sqrt(UH / species["ionization_energy_eV"][0]) - 1.0
            dt = (xi_max - xi_min) / (n_xi * ct.c)

            for j in range(self.ion_atomic_number[i]):
                Uion = species["ionization_energy_eV"][j]
                n_eff = (j + 1) * np.sqrt(UH / Uion)
                C2 = np.pow(2, 2 * n_eff) / (
                    n_eff * math.gamma(n_eff + l_eff + 1.0) * math.gamma(n_eff - l_eff)
                )
                # Note: different factors calculated for ADK and PPT
                if model == "ADK":
                    self.prefactors[i, j, 0] = -(2.0 * n_eff - 1.0)
                    self.prefactors[i, j, 1] = (
                        dt
                        * wa
                        * C2
                        * (Uion / (2.0 * UH))
                        * np.pow(2 * np.pow(Uion / UH, 3.0 / 2.0) * Ea, 2 * n_eff - 1)
                    )
                    self.prefactors[i, j, 2] = (
                        -2.0 / 3.0 * np.pow(Uion / UH, 3.0 / 2.0) * Ea
                    )
                    self.prefactors[i, j, 3] = (
                        (3.0 / ct.pi) * np.pow(Uion / UH, -3.0 / 2.0) / Ea
                    )
                elif model == "PPT":
                    # PPT's prefactors require omega0 which isn't necessarily known yet.
                    # Stash omega0-independent quantum-defect numbers (Ip_au, n_eff, Cnl2).
                    # Finish self.prefactors the first time _calculate_wakefield runs.
                    Ip_au = Uion / AU_ENERGY_EV
                    self.prefactors[i, j, 0] = Ip_au
                    self.prefactors[i, j, 1] = n_eff
                    self.prefactors[i, j, 2] = C2

        self.ion_start_index = (
            np.cumsum(self.ion_atomic_number + 1) - self.ion_atomic_number - 1
        )
        self.ion_densities = np.zeros(
            (np.sum(self.ion_atomic_number + 1), self.n_xi + 4, self.n_r + 4)
        )
        self.elec_density = np.zeros((self.n_xi + 4, self.n_r + 4))

    def _build_ppt_prefactors(self, omega0):
        """
        Fill self.prefactors for the PPT model, given the (now known)
        laser angular frequency omega0.

        PPT's field-dependence enters through the Keldysh parameter
        gamma = omega0*sqrt(2*Ip)/E (dimensionless), with this term
        appearing in arcsinh(gamma), sqrt(1+gamma**2), and a ponderomotive
        shift Up = Ip/(2*gamma**2), which is itself within an exponent. 
        
        ADK is exactly the gamma -> 0 limit of PPT, where all terms simplify
        to the ADK power law/exponential. PPT cannot be simplified so neatly;
        instead we pull out every gamma-independent term (see constants below)
        and use these alongside each evaluation of gamma in do_grid_ionization.

            gamma   = K_gamma / Ep
            kappa   = 2*Ip_au / omega0_au
            ns_exp  = 2*n_eff - 1.5
            C_level = 4*sqrt(2)/pi * Cnl2 * kappa**ns_exp * Ip_au / AU_TIME

        Are the prefactors for PPT in this case.
        """
        omega0_au = omega0 * AU_TIME
        n_species, n_lev = self.prefactors.shape[:2]
        for i in range(n_species):
            for j in range(n_lev):
                Ip_au = self.prefactors[i, j, 0]
                n_eff = self.prefactors[i, j, 1]
                Cnl2  = self.prefactors[i, j, 2]
                if Ip_au == 0.0:
                    continue  # unused slot (species with fewer levels)
                kappa = 2 * Ip_au / omega0_au
                ns_exp = 2 * n_eff - 1.5
                K_gamma = omega0_au * np.sqrt(2 * Ip_au) * AU_EFIELD
                C_level = (
                    4 * np.sqrt(2) / np.pi * Cnl2 * kappa**ns_exp * Ip_au / AU_TIME
                )
                self.prefactors[i, j, 0] = K_gamma
                self.prefactors[i, j, 1] = kappa
                self.prefactors[i, j, 2] = ns_exp
                self.prefactors[i, j, 3] = C_level
        self._ppt_prefactors_ready = True

    def _calculate_wakefield(self, bunches):

        ProfStart("LaserGridIonization")

        # Use a reference plasma density for internal units to be able to simulate vacuum
        self.n_p = 1e23

        if self.laser is None:
            raise ValueError("Must use a laser with LaserGridIonization.")

        # Get laser envelope
        a_env = self.laser.get_envelope()
        is_linear_pol = self.laser.polarization == "linear"

        if self.model == "PPT":
            if not self._ppt_prefactors_ready:
                omega0 = 2 * ct.pi * ct.c / self.laser.l_0
                self._build_ppt_prefactors(omega0)

        density_elec = self.density_function(self.t * ct.c, self.r_fld)
        if self.r_max_plasma is not None:
            density_elec = np.where(self.r_fld > self.r_max_plasma, 0, density_elec)

        # Convert to normalized units and set initial electron density
        self.elec_density[:, :] = 0
        self.elec_density[-2, 2:-2] = density_elec / self.n_p

        self.ion_densities[:, :, :] = 0
        for i, density in enumerate(self.initial_ion_densities):
            density_ion = density(self.t * ct.c, self.r_fld)
            if self.r_max_plasma is not None:
                density_ion = np.where(self.r_fld > self.r_max_plasma, 0, density_ion)
            self.ion_densities[self.ion_start_index[i], -2, 2:-2] = (
                density_ion / self.n_p
            )

        omega0 = 2 * ct.pi * ct.c / self.laser.l_0

        elec_density = self.elec_density[2:-1, 2:-2]

        do_grid_ionization(
            self.model,
            len(self.ion_species),
            elec_density,
            self.ion_densities,
            self.chi,
            a_env,
            self.ion_start_index,
            self.ion_atomic_number,
            self.ion_mass,
            omega0,
            self.prefactors,
            is_linear_pol,
            self.n_xi,
            self.n_r,
            1 / np.abs(self.xi_fld[1] - self.xi_fld[0]),
        )

        if self.species_rho_diags:
            # Gamma for particles with no momentum but that see a laser
            # If linearly polarized, divide by 2 so that the
            # ponderomotive force on the plasma particles is correct.
            gamma_elec = 0.5 * (2 + np.abs(a_env) ** 2 * (0.5 if is_linear_pol else 1))
            self.rho_e[2:-2, 2:-2] = self.elec_density[2:-2, 2:-2] * gamma_elec

            for i in range(len(self.ion_atomic_number)):
                for j in range(self.ion_atomic_number[i] + 1):
                    self.rho_i[2:-2, 2:-2] -= (
                        j * self.ion_densities[self.ion_start_index[i] + j, 2:-2, 2:-2]
                    )

        ProfStop("LaserGridIonization")
