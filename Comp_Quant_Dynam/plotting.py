import numpy as np
import matplotlib.pyplot as plt


#################### Exercise sheet 1 ####################


def plot_func(func, k):
    """
    Plots the function 'func' for a given value of 'k'.
    Useful for creating interactive plots in Jupyter notebooks.
    """

    plt.figure(2)
    plt.clf()
    x = np.linspace(0, 2*np.pi, num=1000)
    plt.plot(x,func(x, k))
    plt.xlabel('x')
    plt.ylabel('func(x, k)')

    return plt.gcf()


#################### Solution sheet 1 ####################


def plot_eigenstate(n, x, evals, evecs):
    """
    Plots the 'n'-th eigenstate of an operator in the position-space representation for a given grid 'x', eigenvalues 'evals', and eigenvectors 'evecs'.
    """
    
    dx = x[1] - x[0]
    evec_amp_n = np.abs(evecs[:,n]) ** 2 / dx
    sig_digits = 5 # number of significant digits to display in the title
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, evec_amp_n)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.6)
    # add labels and legends
    ax.set_title(f"$E_n={evals[n]:.{sig_digits}f}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\\phi_n(x)|^2$")
    
    return fig


#################### Solution sheet 2 ####################


def animate(t, *args):
    """
    Animates the time evolution of a function for a given time 't' and function arguments 'args'.
    The first argument in `args` is the function to be animated, the second argument is the x values for plotting,
    and the third argument is the line object to be updated. The remaining arguments are passed to the function being animated.
    """
    func = args[0] # function to be animated
    xvals = args[1] # x values for plotting
    line = args[2] # line object to be updated
    fargs = args[3:] # function arguments
    y = func(t, *fargs)
    line.set_data(xvals, y)
    return line,


#################### Solution sheet 3 ####################


def multi_animate(t, *args):
    """
    Animates multiple functions for a given time 't' and function arguments 'args'.
    Each function is animated by `animate`, and the results are collected in a list of line objects to be returned.
    The arguments in `args` are expected to be in the format (func, xvals, line, *fargs) for each function to be animated, where `func` is the function to be animated,
    `xvals` are the x values for plotting, `line` is the line object to be updated, and `fargs` are the function arguments.
    """
    
    n_lines = len(args)
    lines = []
    for i in range(n_lines):
        line = animate(t, *args[i])
        lines.append(line)
    return lines


##################### Exercise sheet 4 ####################


def plot_prob_amplitude_2D(t, wfcts, tvec, L):
    """
    Plots the probability amplitude of a 2D wavefunction `wfcts` at time `t` for a given time vector `tvec` and spatial extent `L`.
    The wavefunction is expected to be defined on a grid of size (len(tvec), npoints, npoints), where `npoints` is the number of spatial grid points in each dimension.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(np.abs(wfcts[t]) ** 2, extent=(-L/2, L/2, -L/2, L/2), interpolation='none', origin='lower')
    # add labels and legends
    ax.set_title("$t=$" + str(tvec[t]))
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    plt.show()