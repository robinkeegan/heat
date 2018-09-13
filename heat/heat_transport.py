import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from heat.hydro_funcs import vt_, vs_, hatch_alpha
import pandas as pd


class BP:
    r"""
    Initiates the Bredehoeft and Papaopulos (1965) solution model class.

    Args:

    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param k: thermal conductivity (W/m/C)
    """

    def __init__(self, PwCw, k):
        self.PwCw = PwCw
        self.k = k

    def equation(self, q, To, Tl, z):
        r"""
        Bredehoeft and Papaopulos (1965) solution for one dimensional steady \
        state heat transport.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param To: temperature at z minimum (C)
        :param Tl: temperature at z maximum (C)
        :param z: array of depths (m)
        :returns: temperature at depth/s z

        This is computed using the equation:

        .. math::
               T(z) = To+(Tl-To)(\frac{e^{Ph\frac{z}{L}}-1}{e^{Ph}-1})

        Where:

        .. math::
               Ph = \frac{PwCwqzwL}{k}


        """
        z = z - z.min()
        L = z.max() - z.min()
        z = z[1:-1]
        Ph = (self.PwCw * q * L) / self.k
        t_z = To + (Tl - To) * ((np.exp(Ph * z / L) - 1) / (np.exp(Ph)-1))
        return t_z

    def objective(self, q, To, Tl, z, T):
        r"""
        An objective function which calculates the absolute error between a \
        modelled and observed profile for a flux estimate.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param To: temperature at z minimum (C)
        :param Tl: temperature at z maximum (C)
        :param z: array of depths (m)
        :param T: temperature at z (C)
        :return: absolute error between modelled T(z) and observed T at z.

        """
        return (np.abs(self.equation(q, To, Tl, z) - T)).sum()

    def optimise(self, To, Tl, z, T):
        r"""
        When q is unknown this function will estimate optimal q value for a \
        given profile.

        Args:

        :param To: temperature at z minimum (C)
        :param Tl: temperature at z maximum (C)
        :param z: array of depths (m)
        :param T: temperature at z (C)
        :return: an estimate of q

        """
        result = optimize.minimize(
            self.objective, (1e-10), (To, Tl, z, T),
            tol=1e-10, method="Nelder-Mead",
            options={'maxiter': 1000}
        )
        return result

    def solution(self, T, z, n=3):
        r"""
        Solve analytically for q between a moving boundary.

        Args:

        :param T: array of temperatures (C)
        :param z: array of depths (m)
        :param PwCw: volumetric heat capacity of water (J/m3C)
        :param k: thermal conductivity (W/m/C)
        :return: Temperature at z

        This is computed with:

        .. math::
            q = \frac{k \log{\left (\frac{Tl^{2} - 2.0 Tl Tz + Tz^{2}}{To^{2} - 2.0 To Tz + Tz^{2}} \right )}}{L PwCw}

        """
        func = interp1d(z, T)
        z_new = np.linspace(z.min(), z.max(), n)
        T_new = func(z_new)
        L = (z_new[2:] - z.min())
        To = T_new[0]
        Tz = T_new[1:-1]
        Tl = T_new[2:]
        q_estimates = self.k * np.log((Tl ** 2 - 2.0 * Tl * Tz + Tz ** 2) /
                                      (To ** 2 - 2.0 * To * Tz + Tz ** 2)
                                      ) / (L * self.PwCw)
        return q_estimates


class TurcotteSchubert:
    r"""
    The Turcotte Schubert steady state one dimensional heat transport \
    equation. This is only suitable for upwelling it is unclear what \
    direction q is positive in.

    Args:

    :param Tl: The temperature at L where L is a depth where dT/dz = 0
    :param To: The temperature at z = 0
    :param PwCw: The volumetric heat capacity of water (J/m3C)
    :param k: The thermal conductivity (W/m/C)
    :param z: The depth (m)

    """

    def __init__(self, Tl, To, PwCw, k, z):
        self.Tl = Tl
        self.To = To
        self.PwCw = PwCw
        self.k = k
        self.z = z

    def tz(self, q):
        r"""
        Calculate the temperature at an arbitrary depth.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :return: temperature at z

        Computed using the equation:

        .. math::
            T_z = (T_o - T_l) \cdot e^{\frac{- q \cdot \rho _w c_w \cdot z}{k}} + T_l

        """
        return (self.To - self.Tl) * np.exp(-q * self.PwCw / self.k * self.z) \
            + self.Tl

    def q(self, Tz):
        r"""
        Calculate the flux values.

        Args:

        :param Tz: The temperature at z (C)
        :return: The groundwater flux in m/s

        .. math::
            \frac{-k}{\rho_w c_w \cdot z} \cdot \log{\Big(\frac{T_z-T_l}{T_o - T_l}\Big)}

        """
        return -self.k / (self.PwCw * self.z) * np.log((Tz - self.Tl) / (self.To - self.Tl))


class Stallman:
    r"""
    The Stallman model class

    Args:

    :param PwCw: Volumetric heat capacity of water (J/m3 C)
    :param tau: Period of oscillation (s)
    :param K: The effective thermal conductivity (W/m C)
    :param ne: effective porosity (unit-less)
    :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C)
    :param z: Positive depth below surface where z > 0 and z < infinity.
    :param t: Time/s for evaluation can be a value or a numpy array (s)
    :return: an instance of the Stallman class

    """

    def __init__(self, PwCw, tau, K, ne, pc, z, t):
        self.PwCw = PwCw
        self.tau = tau
        self.k = K
        self.ne = ne
        self.pc = pc
        self.z = z
        self.t = t

    def constants(self, q):
        r"""
        Stallman Constants for the Stallman (1965) Heat Transport equation

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :return: a list with the stallman constants a and b as the zeroth and first elements.

        This is computed using the equations:

        .. math::
            pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

        .. math::
            C = \frac{\pi \cdot pc}{k \cdot T}

        .. math::
            D = \frac{-q \cdot C_wP_w}{2 \cdot k}

        and A and B are constants calculated from C & D via the following \
        equations:

        .. math::
            a = ((C^2 + \frac{D^4}{4})^{1/2} + \frac{D^2}{2}) ^{1/2} - D

        .. math::
            b = ((C^2 + \frac{D^4}{4})^{1/2} - \frac{D^2}{2}) ^{1/2}

        """
        c = (np.pi * self.pc) / (self.k * self.tau)
        d = (q * self.PwCw) / (2 * self.k)
        a = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 + ((d ** 2) / 2)) ** 0.5 - d
        b = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 - ((d ** 2) / 2)) ** 0.5
        return a, b

    def equation(self, a, b):
        r"""
        The Stallman (1965) heat transport equation arranged for amplitude ratio

        Args:

        :param a: constant a
        :param b: constant b
        :return: amplitude ratio (Ar = Ad / As) where Ad is deep sensor and As is shallow sensor

        This is computed using the equation:

        .. math::
            Ar = e^{-a \cdot z} sin(\frac{2 \cdot \pi \cdot t}{T} - b \cdot z)

        """
        return np.exp(-a * self.z) * np.sin(((2 * np.pi * self.t) / self.tau) - b * self.z)

    def objective(self, q, Ar):
        """
        Objective function to calculate absolute error between observed and\
         modelled amplitude ratio

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param Ar: The observed amplitude ratio (Ar = Ad / As) where Ad is deep sensor and As is shallow sensor
        :return: The absolute error between modelled and observed dTz

        """
        a, b = self.constants(q)
        evaluation = self.equation(a, b)
        ae = (np.abs(Ar - evaluation)).sum()
        return ae

    def optimise(self, Ar):
        """
        Optimisation function to calculate the flux which minimises the result\
         of the objective function.

        Args:

        :param Ar: The observed amplitude ratio (Ar = Ad / As) where Ad is deep sensor and As is shallow sensor
        :return: An estimate of the optimal flux (m/s positive downwards)

        """
        result = optimize.minimize(
            self.objective, (0.00001), (Ar), tol=1e-50, method="Nelder-Mead",
            options={'maxiter': 1000}
        )
        return result


def briggs_extinction_depth(K, Am, Ao, a, vt):
    r"""
    Brigg et al. (2014) method to calculate amplitude extinction depth for a \
    sensor of finite precision.

    Args:

    :param K: effective thermal conductivity (W/m C)
    :param Am: minimum detectable amplitude for sensor precision (C)
    :param Ao: amplitude of the signal at surface (C)
    :param a: Hatch alpha term (see hydro_funcs)
    :param vt: thermal front velocity (see hydro_funcs)
    :return: Ze the extinction depth at which amplitude oscillations will be undetectable to a sensor of finite precision

    This is computed using the equation:

    .. math::
        Z_e = 2 \cdot K_e (\frac{\ln(A_{min}/A_{z=0})}{v_t - \sqrt{a + v_t^2 / 2}})

    """

    return 2 * K * (np.log(Am/Ao)) / (vt - (a + (vt**2 / 2))**0.5)


class NumericalTransport:
    """
    Numerical model class.

    Args:

    :param z: The spacing between each temperature in the array (m).
    :param dt: The time step (s) for help choosing a stable time step
    :param PwCw: volumetric heat capacity of water (J/m3 C)
    :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C)
    :param Ke: The effective thermal conductivity (W/m C)
    :return: An instance of the NumericalTransport class.

    """

    def __init__(self, z, dt, PwCw, pc, Ke):
        self.z = z
        self.dt = dt
        self.PwCw = PwCw
        self.pc = pc
        self.Ke = Ke

    def max_timestep(self):
        r"""
        The maximum stable time step calculated with the equation:

        .. math::
            \Delta T \leq \frac{z^2}{2 \cdot K_e}

        :return: The maximum time step (s)
        """
        return self.z ** 2 / (2 * self.Ke)

    def equation(self, q, T):
        r"""
        Governing heat transport equation.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param T: initial temperature depth profile starting with depth 0 as the zeroth element of the array (C).

        A finite difference implementation of the Diffusion Advection equation:

        .. math::
            T_j^{n + 1} = \Delta t \cdot K_e \cdot \frac{T_{j+1} ^n - 2T_j ^n + T_{j - 1} ^n}{z ^2} - \Delta t \cdot \frac{q \cdot \rho_w c_w}{pc} \cdot \frac{T_{j+1}^n - T_{j - 1} ^n}{2z} + T_j ^n

        """
        return self.dt * self.Ke * (T[2:] - 2 * T[1:-1] + T[0:-2]) / self.z ** 2 - self.dt * (q * self.PwCw) / self.pc * (T[2:] - T[0:-2]) / 2 * self.z + T[1:-1]

    def model(self, q, T, n_iterations, top_bc, bot_bc):
        """
        Runs a finite difference model based on initial conditions and \
        boundary conditions and returns the final \
        profile and the profile at all timesteps.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param T: initial temperature depth profile starting with depth 0 as the zeroth element of the array (C).
        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :returns: A list containing the final temperature depth profile as item zero and a dataframe of the temperature profiles at each time step as the second element. If the iterations are to many it may be better to use the method "model2" which only returns the temperature profile at the final time step.

        """
        T_t = []
        for i in range(n_iterations):
            T[0] = top_bc[i]
            T[-1] = bot_bc[i]
            T[1:-1] = self.equation(q, T)
            T_t.append(T.tolist())

        names = np.linspace(0, (len(T) - 1) * self.z, len(T))
        df_ = pd.DataFrame(T_t)
        df_.columns = names.tolist()
        return T, df_

    def objective(self, q, T, n_iterations, top_bc, bot_bc, observed):
        r"""
        Objective method returns the error between an observed and measured \
        2d array (including boundary conditions).

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param T: initial temperature depth profile starting with depth 0 as the zeroth element of the array (C).
        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :param observed: A 2d observed array for all time steps (including the boundary conditions).
        :returns: The absolute error between the observed and modelled profiles

        """
        evaluation = self.model(q, T, n_iterations, top_bc, bot_bc)
        ae = (np.abs(evaluation[1].values - observed)).sum()
        return ae

    def optimise(self, T, n_iterations, top_bc, bot_bc, observed):
        r"""
        Optimisation method returns an optimal estimaate of q. Note for big \
        models it may be necessary to change the "maxiter" option in the \
        "optimise.minimise" function.

        Args:

        :param T: initial temperature depth profile starting with depth 0 as the zeroth element of the array (C).
        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :param observed: A 2d observed array for all time steps (including the boundary conditions).
        :returns: An estimate of the groundwater flux.

        """
        result = optimize.minimize(
            self.objective, (0.00001),
            (T, n_iterations, top_bc, bot_bc, observed),
            tol=1e-50, method="Nelder-Mead", options={'maxiter': 1000}
        )
        return result

    def model2(self, q, T, n_iterations, top_bc, bot_bc):
        r"""
        Runs a finite difference model based on initial conditions and \
        boundary conditions and returns the final profile. This is faster and \
        uses less ram.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param T: initial temperature depth profile starting with depth 0 asthe zeroth element of the array (C).
        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :return: The final temperature depth profile.

        """
        for i in range(n_iterations):
            T[0] = top_bc[i]
            T[-1] = bot_bc[i]
            T[1:-1] = self.equation(q, T)
        return T


class HatchAmplitude:
    r"""
    The Hatch amplitude ratio method.

    Args:

    :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C)
    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param Ke: The effective thermal diffusivity (W/m C)
    :param dz: Distance between sensors (m)
    :param ne: effective porosity (unit-less)
    :param tau: the period of oscillation (s)
    :return: an instance of the Hatch amplitude class

    """

    def __init__(self, pc, PwCw, Ke, dz, ne, tau):
        self.PwCw = PwCw
        self.pc = pc
        self.PwCw = PwCw
        self.Ke = Ke
        self.dz = dz
        self.ne = ne
        self.tau = tau

    def equation(self, q):
        r"""
        The Hatch amplitude method calculating amplitude as a function of \
        depth and flux.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :returns: The amplitude ratio for a given flux and depth.

        .. math::
            A_r = exp \Bigg( \frac{\Delta z}{2 \dot K_e} \Big( v - \sqrt{\frac{a + v^2}{2}} \Big) \Bigg)

        """
        vs = vs_(q, self.ne)
        vt = vt_(self.PwCw, vs, self.ne, self.pc)
        a = hatch_alpha(vt, self.Ke, self.tau)
        Ar = np.exp(self.dz / (2 * self.Ke) * (vt - np.sqrt((a + vt ** 2) / 2)))
        return Ar

    def objective(self, q, Ar):
        r"""
        Objective function to find the absolute error between calculated and \
        observed amplitudes.

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param Ar: observed amplitude ratio (Ar = Ad / As) where Ad is deep sensor and As is shallow sensor
        :return: the absolute error between the modelled and observed amplitudes

        """
        return np.abs(self.equation(q) - Ar)

    def optimise(self, Ar):
        r"""
        Optimisation function to return the q value which minimises the \
        difference between the observed and calculated \
        amplitudes.

        Args:

        :param Ar: observed amplitude ratio (Ar = Ad / As) where Ad is deep sensor and As is shallow sensor
        :return: The optimal q value

        """
        result = optimize.minimize(
            self.objective, (0.00001), (Ar), tol=1e-50, method="Nelder-Mead",
            options={'maxiter': 1000}
        )
        return result
