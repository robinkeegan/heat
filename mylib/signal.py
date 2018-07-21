import numpy as np

def locate_minima(array, period = 24):
    '''
    Find the origin of the first wave period in an time series.

    Args:

    :param array: A sinusoidal time series in the form of a 1D numpy array

    Kwargs:

    :param period: The period for one ocillation (default = 24)
    :return: The index of the origin of the first oscillation.
    '''
    gradient = np.gradient(array)
    abs_gradient = np.abs(gradient)
    minima = abs_gradient[0:period][gradient[0:period] > 0].min()
    mask = abs_gradient[0:period] == minima
    index = np.linspace(0, period-1, period)[mask]
    return int(index[0])

def filter_amp(array, period = 24):
    '''Creates a zero centred sin wave and amplitudes from an amplitude modulated time-series

    Args:

    :param array: A sinusoidal time series in the form of a 1D numpy array

    Kwargs:

    :param period: The period for one oscillation (default = 24)
    :returns: A list containing amplitudes for each cycle and a sine wave for the entire period.

    Example:

    Create a sine wave::

      import numpy as np
      import matplotlib.pyplot as plt
      x = np.linspace(0,20,20)
      A = 2
      T = 20
      d = 15
      y = A * np.sin(((2*np.pi)/T)*x) + d
      plt.plot(y)
      plt.show()

    Use the function to extract the amplitude::

      amp = filter_amp(y, period = 20)[0]
      print(amp)
      >> [1.99316899]

    Or output the modelled sine wave::

      wave = filter_amp(y, period = 20)[1]
      plt.plot(y, label = 'Original')
      plt.plot(wave, label = 'Modelled')
      plt.legend()
      plt.show()
        '''
    # Equation of a sin wave
    def _y(x, a, T, d = 0):
        y_ = a * np.sin(((2*np.pi)/T)*x) + d
        return y_
    # Equation to extract amplitude
    def amp(y, T):
        b = (2*np.pi)/T
        T_A = (y.max() / (np.sin(b*(T/4)))) - y.mean()
        B_A = (y.min() / (np.sin(b*(3*T/4)))) + y.mean()
        return np.array([T_A, B_A]).mean()
    # Model to process data
    n_periods = int(len(array)/period)
    sin_wave = np.zeros(len(array))
    amplitudes = np.zeros(n_periods)
    x = np.linspace(0, period, period)
    for i in range(n_periods):
        b_ind = i * period
        t_ind = b_ind + period
        amplitudes[i] = amp(array[b_ind:t_ind], period)
        sin_wave[b_ind:t_ind] = _y(x, amplitudes[i], period)
    return [amplitudes, sin_wave]