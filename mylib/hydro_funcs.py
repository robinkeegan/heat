
def pc_(ne, PwCw, PsCs):
    r'''
    Bulk volumetric heat capacity of a saturated medium

    Args:

    :param ne: effective porosity (unit-less)
    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param PsCs: volumetric heat capacity of solid (J/m3C)
    :return: pc the bulk volumetric heat capacity of the saturated medium (J/m3C)

    This is computed using the equation:

    .. math::
        pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

    '''
    return ne * PwCw + (1-ne) * PsCs

def vs_(q, ne):
    r'''
    Solute Front Velocity

    Args:

    :param q: flux (m/s)
    :param ne: effective porosity (unit-less)
    :return: vs the solute front velocity (m/s)

    This is computed with the equation:

    .. math::
        v_s = \frac{q}{n_w}

    '''
    return q / ne

def vt_(PwCw, vs, ne, pc):
    r'''
    Thermal Front Velocity

    Args:

    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param vs: solute front velocity (m/s)
    :param ne: effective porosity (unit-less)
    :param pc: bulk volumetric heat capacity of a saturated medium (J/m3C)
    :return: vt the thermal front velocity

    This is computed with the equation:

    .. math::
        v_t = v_s \cdot \frac{P_wC_w}{pc}

    '''
    return vs * (PwCw/pc) * ne


def vt_full(ne, PwCw, PsCs, q):
    r'''
    Calculate the thermal front velocity without intermediately calculating the solute front velocity and bulk \
    volumetric heat capacity of the saturated medium.

    Args:

    :param ne: effective porosity (unit-less)
    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param PsCs: volumetric heat capacity of solid (J/m3C)
    :param q: flux (m/s)
    :return: vt the thermal front velocity

    This is computed with the equations:

    .. math::
        pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

    .. math::
        v_s = \frac{q}{n_w}

    .. math::
        v_t = v_s \cdot \frac{P_wC_w}{pc}

    '''
    pc = ne * PwCw + (1-ne) * PsCs
    vs = q / ne
    return vs * (PwCw/pc) * ne

def ke_(Kw, Ks, ne, pc):
    r'''
    Effective thermal conductivity

    Args:

    :param Kw: thermal conductivity of water (W/m C)
    :param Ks: thermal conductivity of solids (W/m C)
    :param ne: effective porosity (unit-less)
    :param pc: bulk volumetric heat capacity of a saturated medium (J/m3C)
    :return: ke the effective thermal conductivity (W/m C)

    This is computed with the equation:

    .. math::
        k_e = \frac{k_w^{ne} \cdot k_s ^{(1-ne)}}{pc}

    '''
    return (Kw**ne * Ks ** (1 - ne))/pc

def ke_full(Kw, Ks, ne, PwCw, PsCs):
    r'''
    The effective thermal conductivity without intermediately calculating bulk volumetric heat capacity of a \
    saturated medium

    Args:

    :param Kw: thermal conductivity of water (W/m C)
    :param Ks: thermal conductivity of solids (W/m C)
    :param ne: effective porosity (unit-less)
    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param PsCs: volumetric heat capacity of solid (J/m3C)
    :return: ke the effective thermal conductivity (W/m C)

    This is computed with the equations:

     .. math::
        pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

     .. math::
        k_e = \frac{k_w^{ne} \cdot k_s ^{(1-ne)}}{pc}

    '''
    pc = ne * PwCw + (1-ne) * PsCs
    return (Kw**ne * Ks ** (1 -ne))/pc

def peclet(PwCw, q, L, ke):
    r'''
    The Peclet number for heat transport. Note when Peclet number < 0.5 dispersivity can be neglected (Rau et al. 2012).

    :param PwCw:
    :param q: volumetric heat capacity of water (J/m3C)
    :param L: length of flow path (m)
    :param ke: effective thermal conductivity (W/m C)
    :return: Ph the Peclet number for heat flow

    This is computed with the equation:

    .. math::
        Ph = \frac{P_wC_w \cdot q \cdot L}{k_e}

    '''
    return (PwCw * q * L)/ke
