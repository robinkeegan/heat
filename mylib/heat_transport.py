import numpy as np
from scipy import optimize
from mylib.signal import filter_amp
from mylib.hydro_funcs import vt_, vs_, hatch_alpha

class BP:
    def __init__(self, PwCw, L, k, To, Tl, z):
        '''
        Initiates the Bredehoeft and Papaopulos (1965) solution model class

        Args:

        :param z: depth (m)
        :param L: maximum depth (m)
        :param To: temperature at z = 0 (C)
        :param Tl: temperature at z = L (C)
        :param q: groundwater flux positive downwards (m/s)
        :param PwCw: volumetric heat capacity of water (J/m3C)
        :param k: thermal conductivity (W/m/C)

        '''
        self.PwCw = PwCw
        self.L = L
        self.k = k
        self.To = To
        self.Tl = Tl
        self.z = z

    def equation(self, q):
        r'''
        Bredehoeft and Papaopulos (1965) solution for one dimensional steady state heat transport.

        This is computed using the equation:

        .. math::
               T(z) = To+(Tl-To)(\frac{e^{Ph\frac{z}{L}}-1}{e^{Ph}-1})

        Where:

        - z = depth (m)
        - L = maximum depth (m)
        - To = temperature at z = 0 (C)
        - Tl = temperature at z = L (C)
        - q = q in the z direction (m/s)
        - PwCw = volumetric heat capacity of water (J/m3C)
        - k = thermal conductivity (W/m/C)

        and:

        .. math::
               Ph = \frac{PwCwqzwL}{k}


        '''
        Ph = (self.PwCw * q * self.L) / self.k
        t_z = self.To + (self.Tl - self.To)*((np.exp(Ph * self.z / self.L) - 1)/(np.exp(Ph)-1))
        return t_z

    def objective(self, q, T):
        '''
        An objective function which calculates the absolute error between a modelled and observed profile for a flux \
        estimate.

        :param T: temperature at z (C)
        :return: absolute error between modelled T(z) and observed T at z.
        '''
        return (np.abs(self.equation(q) - T)).sum()

    def optimise(self, T):
        '''
        When q is unknown this function will estimate optimal q value for a given profile.

        :param T: temperature at z (C)
        :return: an estimate of q
        '''
        result = optimize.minimize(self.objective, (0.00001),(T),tol=1e-50, method="Nelder-Mead",
                                   options= {'maxiter': 1000})
        return result

class Stallman:
    '''
    The Stallman model class

    Args:

    :param PwCw: Volumetric heat capacity of water (J/m3 C)
    :param tau: Period of oscillation (s)
    :param Ke: The effective thermal conductivity (W/m C) see hydro_funcs.ke_ for help
    :param ne: effective porosity (unit-less)
    :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C) see hydro_funcs.pc_ for help
    :param dT: The amplitude of the oscillation at z=0 (C) see signal._filter for help extracting amplitude
    :param z: Positive depth below surface where z > 0 and z < infinity.
    :param t: Time/s for evaluation can be a value or a numpy array (s)
    :return: an instance of the Stallman class

    '''

    def __init__(self, PwCw, tau, Ke, ne, pc, dT, z, t):
        self.PwCw = PwCw
        self.tau = tau
        self.k = Ke
        self.ne = ne
        self.pc = pc
        self.dT = dT
        self.z = z
        self.t = t

    def constants(self, q):
        r'''
        Stallman Constants for the Stallman (1965) Heat Transport equation

        :return: a list with the stallman constants a and b as the zeroth and first elements.

        This is computed using the equations:

        .. math::
            pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

        .. math::
            C = \frac{\pi \cdot pc}{k \cdot T}

        .. math::
            D = \frac{-q \cdot C_wP_w}{2 \cdot k}

        and A and B are constants calculated from C & D via the following equations:

        .. math::
            a = ((C^2 + \frac{D^4}{4})^{1/2} + \frac{D^2}{2}) ^{1/2} - D

        .. math::
            b = ((C^2 + \frac{D^4}{4})^{1/2} - \frac{D^2}{2}) ^{1/2}

        '''
        c = (np.pi * self.pc) / (self.k * self.tau)
        d = (q * self.PwCw) / (2 * self.k)
        self.a = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 + ((d ** 2) / 2)) ** 0.5 - d
        self.b = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 - ((d ** 2) / 2)) ** 0.5
        return self.a, self.b

    def equation(self):
        r'''
        The Stallman (1965) heat transport equation


        :return: An amplitude at z

        This is computed using the equation:

        .. math::
            T-T_o = \Delta T \cdot e^{-a \cdot z} sin(\frac{2 \cdot \pi \cdot t}{T} - b \cdot z)

        '''
        return self.dT * np.exp(-self.a * self.z) * np.sin(((2 * np.pi * self.t) / self.tau) - self.b * self.z)

    def objective(self, q, dTz):
        '''

        :param q: groundwater flux positive downwards (m/s)
        :param dTz: The observed amplitude of the oscillation at z (C) see signal._filter for help extracting amplitude
        :return:
        '''
        constants = self.constants(q)
        evaluation = self.equation()
        ae = np.abs(dTz - filter_amp(evaluation)[0])
        return ae

    def optimise(self, dTz):
        '''
        :param dTz: The observed amplitude of the oscillation at z (C) see signal._filter for help extracting amplitude
        :return: An estimate of the optimal flux
        '''
        result = optimize.minimize(self.objective, (0.00001),(dTz),tol=1e-50, method="Nelder-Mead",
                                   options= {'maxiter': 1000})
        return result


def briggs_extinction_depth(ke, Am, Ao, a, vt):
    r'''
    Brigg et al. (2014) method to calculate amplitude extinction depth for a sensor of finite precision.

    :param ke: effective thermal conductivity (W/m C)
    :param Am: minimum detectable amplitude for sensor precision (C)
    :param Ao: amplitude of the signal at surface (C)
    :param a: Hatch alpha term (see hydro_funcs)
    :param vt: thermal front velocity (see hydro_funcs)
    :return: Ze the extinction depth at which amplitude oscillations will be undetectable to a sensor of finite precision

    This is computed using the equation:

    .. math::
        Z_e = 2 \cdot K_e (\frac{\ln(A_{min}/A_{z=0})}{v_t - \sqrt{a + v_t^2 / 2}})

    '''

    return 2 * ke * (np.log(Am/Ao)) / (vt - (a + (vt**2 / 2))**0.5)


class NumericalTransport:

    def __init__(self, T, z, dt, q, PwCw, pc, Ke):
        '''
        Args:

        :param T: initial temperature depth profile starting with depth 0 as the zeroth element of the array (C).
        :param z: The spacing between each temperature in the array (m).
        :param dt: The time step (s) for help choosing a stable time step see method .max_timestep
        :param q: groundwater flux positive downwards (m/s)
        :param PwCw: volumetric heat capacity of water (J/m3 C)
        :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C) see hydro_funcs.pc_ for help
        :param Ke: The effective thermal conductivity (W/m C) see hydro_funcs.ke_ for help
        :return: An instance of the NumericalTransport class.
        '''
        self.T = T
        self.z = z
        self.dt = dt
        self.q = q
        self.PwCw = PwCw
        self.pc = pc
        self.Ke = Ke

    def equation(self):
        '''

        A finite difference implementation of the Diffusion Advection equation:
        .. math::
            T_j^{n + 1} = \Delta t \cdot K_e \cdot \frac{T_{j+1} ^n - 2T_j ^n + T_{j - 1} ^n}{z ^2} - \Delta t \cdot \
            \frac{q \cdot \rho_w c_w}{pc} \cdot \frac{T_{j+1}^n - T_{j - 1} ^n}{2z} + T_j ^n
        '''
        return self.dt * self.Ke * (self.T[2:] - 2 * self.T[1:-1] + self.T[0:-2]) / self.z ** 2 - self.dt * \
            (self.q * self.PwCw) / self.pc * (self.T[2:] - self.T[0:-2]) / 2 * self.z + self.T[1:-1]

    def max_timestep(self):
        '''
        The maximum stable time step calculated with the equation:

        .. math::
            \Delta T \leq \frac{z^2}{2 \cdot K_e}

        :return: The maximum time step (s)
        '''
        return self.z ** 2 / (2 * self.Ke)

    def model(self, n_iterations, top_bc, bot_bc):
        '''
        Runs a finite difference model based on initial conditions and boundary conditions and returns the final \
        profile and the profile at all timesteps.

        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :return: A list containing the final temperature depth profile as item zero and a list of the temperature \
        profiles at each time step as the second element. If the iterations are to many it may be better to use the \
        method model 2 which only returns the temperature profile at the final time step.
        '''
        T_t = []
        for i in range(n_iterations):
            self.T[0] = top_bc[i]
            self.T[-1] = bot_bc[i]
            self.T[1:-1] = self.equation()
            T_t.append(self.T)
        return self.T, T_t

    def model2(self, n_iterations, top_bc, bot_bc):
        '''
        Runs a finite difference model based on initial conditions and boundary conditions and returns the final \
        profile. This is faster and uses less ram.

        :param n_iterations: The number of iterations to run the model (integer)
        :param top_bc: An array of the temperature at the top boundary condition (C) must be of len(n_iterations)
        :param bot_bc: An array of the temperature at the bottom boundary condition (C) must be of len(n_iterations)
        :return: The final temperature depth profile.

        '''
        for i in range(n_iterations):
            self.T[0] = top_bc[i]
            self.T[-1] = bot_bc[i]
            self.T[1:-1] = self.equation()
        return self.T

class HatchAmplitude:

    def __init__(self, pc, PwCw, Ke, dz, Ar, ne, tau):
        '''
        The Hatch amplitude ratio method.

        :param pc: The bulk volumetric heat capacity of the saturated medium (J/m3C) see hydro_funcs.pc_ for help
        :param PwCw: volumetric heat capacity of water (J/m3C)
        :param Ke: The effective thermal conductivity (W/m C) see hydro_funcs.ke_ for help
        :param z: Depth (m)
        :param Ar: Ar = Ad / As where A_d is deep sensor and A_s is shallow sensor
        :param ne: effective porosity (unit-less)
        :param tau: the period of oscillation (s)
        :return: an instance of the Hatch amplitude class
        '''
        self.PwCw = PwCw
        self.pc = pc
        self.PwCw = PwCw
        self.Ke = Ke
        self.dz = dz
        self.Ar = Ar
        self.ne = ne
        self.tau = tau

    def equation(self, q):
        r'''
        The Hatch amplitude method computed with the equation:

        .. math::
            qA_r = \frac{pc}{\rho_w C_w} \cdot \Bigg(\frac{2 \cdot K_e}{\Delta z} \cdot ln(A_r) + \sqrt{\frac{a + \
            v_t^2}{2}}\Bigg)

        '''
        vs = vs_(q, self.ne)
        vt = vt_(self.PwCw, vs, self.ne, self.pc)
        a = hatch_alpha(vt, self.Ke, self.tau)
        return np.exp((self.dz / (2 * self.Ke)) * (vt - np.sqrt((a + vt ** 2) / 2)))

    def objective(self, q, Ak):
        '''
        Objective function to find the absolute error between calculated and observed amplitudes.

        :param q: groundwater flux positive downwards (m/s)
        :param Ak: observed amplitude ratio
        :return: the absolute error between the modelled and observed amplitudes
        '''
        return np.abs(self.equation(q) - Ak)

    def optimise(self, Ak):
        '''
        Optimisation function to return the q value which minimises the difference between the observed and calculated \
        amplitudes.
        :param Ak: observed amplitude ratio
        :return: The optimal q value
        '''
        result = optimize.minimize(self.objective, (0.00001),(Ak), tol=1e-50, method="Nelder-Mead",
                                   options= {'maxiter': 1000})
        return result

