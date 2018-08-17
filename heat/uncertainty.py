def mc(x1, x2):
    r"""
    Find all posible combinations of two ranges.

    Args:

    :param x1: Upper and lower intervals for paramter x1 as list.
    :param x2: Upper and lower intervals for paramter x2 as list.
    :returns: A two dimensional list of all possible combinations of two ranges.
    """
    return [[[j, k] for j in x1] for k in x2]
