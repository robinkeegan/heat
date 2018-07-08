
def b_p(qzw, Pw, Cw, L, k, To, Tl, z):
    r'''
    This equation computes the BP solution for one dimensional heat transport.

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
           Ph = \frac{PwCwqzwL}{k}


    '''
    Ph = (Pw * Cw * qzw * L)/k
    t_z = To + (Tl-To)*((np.exp(Ph*z/L)-1)/(np.exp(Ph)-1))
    return(t_z)