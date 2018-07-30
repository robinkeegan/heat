import numpy as np
from statsmodels.tsa.stattools import ccf
from scipy import signal
from statsmodels.tsa.tsatools import detrend


def amplitude(x, y, period=24, tol=0.5, nout=10000):
    '''
    Extract the amplitude amplitude from a time-series using the lombscargle method.

    Args:

    :param x: time component
    :param y: the wave at each time-step
    :param period: the period of one oscillation
    :param tol: how far above and below the period to compute the periodigram default = 0.5 * period
    :param nout: number of periods to output
    :returns: Periods, the associated amplitudes for each period, the index of the max amplitude and the max amplitude.
    '''
    y = detrend(y)
    tolerance = period * tol
    periods = np.linspace(period - tolerance, period + tolerance, nout)
    freqs = 1 / periods
    angular_freqs = 2 * np.pi * freqs
    pgram = signal.lombscargle(x, y, angular_freqs)
    normalized_pgram = np.sqrt(4 * (pgram / len(y)))
    index = np.argmax(normalized_pgram)
    return periods, normalized_pgram, index, normalized_pgram[index]


def get_amp(array):
    '''
    Applys the amplitude function to get amean amplitude over a 7 day period.

    Args:

    :param array: array of an hourly time series
    :returns: mean amplitude for each 7 day period
    '''
    week = 24 * 7
    nweeks = int(len(array) / week)
    output = np.zeros(nweeks)
    for i in range(nweeks):
        bind = i * week
        tind = bind + week
        subset = array[bind:tind]
        output[i] = amplitude(np.linspace(0, len(subset) - 1, len(subset)), subset)[3]
    return output


def phase_offset(y1, y2, period=24):
    '''
    Find the lag or offset between time-series y1 and y2 using cross-correlation.

    Args:

    :param y1: Time series 1
    :param y2: Time series 2
    :param period: The period of one oscillation default = 24
    :return: the lag between time series 1 and 2, and the index of the max correlation.
    '''
    correlation = ccf(y2, y1)
    index = np.argmax(correlation[0:int(2 * period)])
    return correlation, index

