import numpy as np
from scipy import optimize


class b_p:
    def __init__(self, q, PwCw, L, k, To, Tl, z):
        '''

        Args:

        :param z: depth (m)
        :param L: maximum depth (m)
        :param To: temperature at z = 0 (C)
        :param Tl: temperature at z = L (C)
        :param q: groundwater flux positive downwards (m/s)
        :param PwCw: volumetric heat capacity of water (J/m3C)
        :param k: thermal conductivity (W/m/C)

        '''
        self.q = q
        self.PwCw = PwCw
        self.L = L
        self.k = k
        self.To = To
        self.Tl = Tl
        self.z = z

    def equation(self):
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
        Ph = (self.PwCw * self.q * self.L) / self.k
        t_z = self.To + (self.Tl - self.To)*((np.exp(Ph * self.z / self.L) - 1)/(np.exp(Ph)-1))
        return t_z

    def objective(self, T):
        '''
        An objective function which calculates the absolute error between a modelled and observed profile for a flux
        estimate.

        :param T: temperature at z (C)
        :return: absolute error between modelled T(z) and observed T at z.
        '''
        return (np.abs(self.equation(self) - T)).sum()

    def optimise(self, T):
        '''
        When q is unknown this function will estimate optimal q value for a given profile.

        :param T: temperature at z (C)
        :return: an estimate of q
        '''
        result = optimize.minimize(self.objective, (0.000001),(self.Pw, self.Cw, self.L, self.k, self.To, self.Tl, self.z, T),
                                   tol=1e-50, method="Nelder-Mead",  options= {'maxiter': 1000})
        return result.x[0]


class Stallman:
    '''The Stallman model class has the option to calculate the Stallman constants or run the Stallman equation.
    '''

    def constants(q, PsCs, PwCw, T, k, ne):
        r'''
        Stallman Constants for the Stallman (1965) Heat Transport equation

        Args:

        :param q: groundwater flux positive downwards (m/s)
        :param PsCs: Volumetric heat capacity of the matrix (J/m3 C)
        :param PwCw: Volumetric heat capacity of water (J/m3 C)
        :param T: Period of oscillation (s)
        :param k: Thermal conductivity (W/m/C)
        :param ne: effective porosity (unit-less)
        :return: A list with the stallman constants a and b as the zero'th and first elements.

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
        pc = ne * PwCw + (1 - ne) * PsCs
        c = (np.pi * pc) / (k * T)
        d = (q * PwCw) / (2 * k)
        a = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 + ((d ** 2) / 2)) ** 0.5 - d
        b = ((c ** 2 + ((d ** 4) / 4)) ** 0.5 - ((d ** 2) / 2)) ** 0.5
        return [a, b]

    def stallman(dT, a, z, b, t, T):
        r'''
        The Stallman (1965) heat transport equation

        Args:

        :param dT: The amplitude of the oscillation at z=0 (C)
        :param a: Stallman constant 'a' from function 'stallman_cons'
        :param z: Positive depth below surface where z > 0 and z < infinity.
        :param b: Stallman constant 'b' from function 'stallman_cons'
        :param t: Time/s for evaluation can be a value or a numpy array (s)
        :param T: Period of oscillation (s)
        :return: An amplitude at depth

        This is computed using the equation:

        .. math::
            T-T_o = \Delta T \cdot e^{-a \cdot z} sin(\frac{2 \cdot \pi \cdot t}{T} - b \cdot z)

        '''
        amp = dT * np.exp(-a * z) * np.sin(((2 * np.pi * t) / T) - b * z)
        return amp


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

    def __int__(self, T, z, dt, q, PwCw, pc, Ke):
        '''
        Args:

        :param T:
        :param z:
        :param dt:
        :param q:
        :param PwCw:
        :param pc:
        :param Ke:
        :return:
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

        '''
        return self.dt * self.Ke * (self.T[2:] - 2 * self.T[1:-1] + self.T[0:-2]) / self.z ** 2 - self.dt * \
            (self.q * self.PwCw) / self.pc * (self.T[2:] - self.T[0:-2]) / 2 * self.z + self.T[1:-1]

    def max_timestep(self):
        return self.z ** 2 / (2 * self.Ke)

    def max_gridspace(self):
        return (2 * self.Ke) / ((self.q * self.PwCw) / self.pc)

    def model(self, n_iterations, top_bc, bot_bc):
        '''
        :param n_iterations:
        :param top_bc:
        :param bot_bc:
        :return:
        '''
        lis = []
        for i in range(n_iterations):
            self.T[0] = top_bc[i]
            self.T[-1] = bot_bc[i]
            self.T[1:-1] = self.equation(self)
            lis.append(self.T)
        return self.T, lis

