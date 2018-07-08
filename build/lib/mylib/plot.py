import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
plt.style.use('seaborn-white')


def contour(x, y, z, xlab = '', ylab = '', cblab = '', size = (6,4), save = False, fname = None, cmap = "OrRd"):
    '''Grid three one dimensional numpy arrays of equal length and create a contour plot

    Args:

    :param x: A one dimensional numpy array

    :param y: A one dimensional numpy array

    :param z: A one dimensional numpy array

    :returns: A contour plot.

    Kwargs:

    :param xlab: A string x label (default= '')
    :param ylab: A string x label (default= '')
    :param figsize: figure size in tuple x, y in size inches (default = (6,4))
    :param marker: Show markers True or False (default = True)
    :param save: Save the figure True or False (default = False)
    :param fname: If save is True specify a filename (default = None)
    :param cmap: The colour map (default = "OrRd")
      '''
    # Mesh the grid
    yi, xi = np.mgrid[int(y.min()):int(y.max()), int(x.min()):int(x.max())]
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # Create the figure
    plt.figure(figsize=size)
    plt.imshow(zi,extent=[x.min(), x.max(), y.min(), y.max()],
               origin="lower", interpolation='bicubic', aspect='auto',cmap=cmap)
    plt.colorbar().set_label(cblab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # Save or show the plot
    if save == True:
        if fname == None:
            print('Error if saving a filename must be specified')
        else:
            plt.savefig(fname)
    if save == False:
        plt.show()


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

