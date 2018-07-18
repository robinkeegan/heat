import numpy as np

def b_p(qzw, Pw, Cw, L, k, To, Tl, z):
    r'''
    Bredehoeft and Papaopulos (1965) solution for one dimensional steady state heat transport.

    Args:

    :param z: depth (m)
    :param L: maximum depth (m)
    :param To: Temperature at z = 0 (C)
    :param Tl: Temperature at z = L (C)
    :param qzw: q in the z direction (m/s)
    :param pw: density of water (1000kg/m3)
    :param Cw: specific heat capacity of water (4184 joule/kg C)
    :param k: thermal conductivity (W/m/C)
    :returns T(z): The temperature at z = n  where n >0 and n< infinity.

    This is computed using the equation:

    .. math::
           T(z) = To+(Tl-To)(\frac{e^{Ph\frac{z}{L}}-1}{e^{Ph}-1})

    Where:

    - z = depth (m)
    - L = maximum depth (m)
    - To = Temperature at z = 0 (C)
    - Tl = Temperature at z = L (C)
    - qzw = q in the z direction (m/s)
    - pw = density of water (1000kg/m3)
    - Cw = specific heat capacity of water (4184 joule/kg C)
    - k = thermal conductivity (W/m/C)

    and:

    .. math::
           Ph = -\frac{PwCwqzwL}{k}


    '''
    Ph = -(Pw * Cw * qzw * L)/k
    t_z = To + (Tl-To)*((np.exp(Ph*z/L)-1)/(np.exp(Ph)-1))
    return(t_z)


def stallman_cons(q, PsCs, PwCw, T, k, ne):
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
    pc = ne * PwCw + (1-ne) * PsCs
    c = (np.pi * pc) / (k * T)
    d = (-q * PwCw) / (2 * k)
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
    :return: Ze the extinction depth at which amplitude oscillations will be undetectable to a sensor of finite \
    precision

    This is computed using the equation:

    .. math::
        Z_e = 2 \cdot K_e (\frac{\ln(A_{min}/A_{z=0})}{v_t - \sqrt{a + v_t^2 / 2}})

    '''

    return 2 * ke * (np.log(Am/Ao)) / (vt - (a + (vt**2 / 2))**0.5)
