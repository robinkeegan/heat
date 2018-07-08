import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


def contour(x, y, z):
    '''Grid three one dimensional numpy arrays of equal length and create a contour plot

    Args:

    :param x: A one dimensional numpy array

    :param y: A one dimensional numpy array

    :param z: A one dimensional numpy array

    :returns: A contour plot.

    Example::

        import numpy as np
        x = np.linspace(1,10,1)
        y = x
        z = x
        contour(x,y,z)
      '''
    pass


def plot(x,y, xlab = '', ylab = '', c = 'k', figsize = (6,4), marker = True, save = False, fname = None):
    '''
    Plot a x and y on a graph.

    Args:

    :param x: A list or array of values.
    :param y: A list or array of values.

    Kwargs:

    :param xlab: A string x label (default= '')
    :param ylab: A string x label (default= '')
    :param c: A string colour (default= 'k')
    :param figsize: figure size in tuple x, y in size inches (default = (6,4))
    :param marker: Show markers True or False (default = True)
    :param save: Save the figure True or False (default = False)
    :param fname: If save is True specify a filename (default = None)
    :return: Either shows a plot or saves a plot.
    '''
    plt.figure(figsize =figsize)
    #Marker or not
    if marker == True:
        plt.plot(x,y, c= c, marker ='o')
    elif marker == False:
        plt.plot(x,y, c= c)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #Show or save figure
    if save == True:
        if fname == None:
            print("Error if saving a filename must be specified")
        else:
            plt.tight_layout()
            plt.savefig(fname)
    elif save == False:
        plt.show()

