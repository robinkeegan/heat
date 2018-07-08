from scipy import stats
import numpy as np

def r2(x,y):
    '''
    Return an R2 value for two arrays of equal length.

    Args:

    :param x: The independent variable.
    :param y: The dependent variable.
    :return: The coefficient of determination.
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

def linear_eq(x,y):
    '''
    Find the equation of a line.

    Args:

    :param x: The independent variable.
    :param y: The coefficient of determination.
    :returns: A the function y = f(x), the slope, the intercept of the line and the R2 of the line to the original data.
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    def f(x, slope, intercept):
        return x * slope + intercept
    fit = f(x, slope, intercept)
    return [fit, slope, intercept, r2(x, fit)]

