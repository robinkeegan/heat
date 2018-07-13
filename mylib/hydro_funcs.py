
def pc_(ne, PwCw, PsCs):
    r'''
    Calculate the bulk volumetric heat capacity of a saturated medium.
    Args:

    :param ne: is effective porosity (unitless)
    :param PwCw: volumetric heat capacity of water (J/m3C)
    :param PsCs: volumetric heat capacity of solid (J/m3C)
    :return: pc the bulk volumetric heat capacity of the saturated medium (J/m3C)

    This is computed using the equation:

    .. math::
        pc = n_e \cdot P_wC_w + (1 - n_e) \cdot P_sC_s

    '''
    return ne * PwCw + (1-ne) * PsCs

