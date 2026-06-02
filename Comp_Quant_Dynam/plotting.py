import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp
from scipy.sparse import linalg as sLA
import Comp_Quant_Dynam.utility as utility
import Comp_Quant_Dynam.operators as operators


#################### Exercise sheet 1 ####################


def plot_func(func, k):
    """
    Plots the function 'func' for a given value of 'k'.
    Useful for creating interactive plots in Jupyter notebooks.
    """

    plt.figure(2)
    plt.clf()
    x = np.linspace(0, 2 * pi, num=1000)
    plt.plot(x, func(x, k))
    plt.xlabel('x')
    plt.ylabel('func(x, k)')

    return plt.gcf()


#################### Solution sheet 1 ####################


def plot_eigenstate(n, x, evals, evecs):
    """
    Plots the 'n'-th eigenstate of an operator in the position-space representation for a given grid 'x', eigenvalues 'evals', and eigenvectors 'evecs'.
    """
    
    dx = x[1] - x[0]
    evec_amp_n = np.abs(evecs[:, n]) ** 2 / dx
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
    ext = L / 2
    ax.imshow(np.abs(wfcts[t]) ** 2, extent=(-ext, ext, -ext, ext), interpolation='none', origin='lower')
    # add labels and legends
    ax.set_title("$t=$" + str(tvec[t]))
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    plt.show()


##################### Exercise sheet 6 ####################


def plot_compare_ED(N, observables_Integrator, observables_ED, tvec_output):
    """
    Plots the expectation values of the observables calculated with the numerical integrator and the exact diagonalization results for a system of size `N` at the times given by `tvec_output`.
    The observables are expected to be stored in the first dimension of the `observables_Integrator` and `observables_ED` arrays, and the time evolution is expected to be stored in the second dimension.
    """
    plt.plot(tvec_output, observables_ED[0] / N,'k--')
    plt.plot(tvec_output, observables_Integrator[0] / N)
    plt.ylim([-1,1])
    plt.xlim([0,tvec_output[-1]])
    plt.xlabel('$t$')
    plt.ylabel('$S_z/N$')
    plt.legend(['exact diagonalization','num. integrator'])
    plt.show()

    plt.plot(tvec_output, observables_ED[1] / N, 'k--')
    plt.plot(tvec_output, observables_Integrator[1] / N)
    plt.ylim([-1, 1])
    plt.xlim([0, tvec_output[-1]])
    plt.xlabel('$t$')
    plt.ylabel('$S_x/N$')
    plt.legend(['exact diagonalization','num. integrator'])

    plt.show()

    plt.plot(tvec_output, sqrt(observables_ED[2]) - 1, 'k--')
    plt.plot(tvec_output, sqrt(observables_Integrator[2]) - 1)
    # plt.ylim([0,2])
    plt.xlabel('t')
    plt.ylabel('norm-1')
    plt.show()

def plot_deviations(idx_stepsize, deviations, tvec_output, step_sizes):
    """
    Plots the deviations of the observables Sz, Sx, and the norm of the state from the exact diagonalization results for a given time step size index `idx_stepsize`,
    deviations array `deviations`, output time vector `tvec_output`, and array of time step sizes `step_sizes`.
    """
    
    plt.plot(tvec_output, deviations[idx_stepsize, 0], 'k')
    plt.title('time step for Integrator: '+ str(step_sizes[idx_stepsize]))
    plt.xlabel('t')
    plt.ylabel('$S_z$ absolute error')
    plt.show()

    plt.plot(tvec_output, deviations[idx_stepsize, 1], 'k')
    plt.title('time step for Integrator: '+ str(step_sizes[idx_stepsize]))
    plt.xlabel('t')
    plt.ylabel('$S_x$ absolute error')
    plt.show()

    plt.plot(tvec_output, np.sqrt(deviations[idx_stepsize, 2]), 'k')
    plt.title('time step for Integrator: '+ str(step_sizes[idx_stepsize]))
    plt.xlabel('t')
    plt.ylabel('norm error')
    plt.show()


##################### Exercise sheet 7 ####################

def rotation_bloch_sphere(phi, ini, operator, bloch):
    # rotate by an angle phi 
    # (using expm for convenience - You could also calculate the component vector after a general rotation analytically)
    sx = operators.sigma_x_sparse() / 2
    sy = operators.sigma_y_sparse() / 2
    sz = operators.sigma_z_sparse() / 2
    spin_obsv = [sx, sy, sz]
    state = sLA.expm(-1j * phi * operator) @ ini
    # calculate the spin expectation values
    spin_comps = utility.expectation_value(state, spin_obsv)
    # update the existing sphere
    bloch.clear()
    bloch.add_points(2 * spin_comps)
    bloch.show()

def plot_H_all(Hfront, Hback, Htop, grid, HscaleMax = 1.0):
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 3, 1) # front
    plt.imshow(Hfront, extent=(-1, 1, -1, 1), aspect=1)
    plt.contourf(grid, grid, Hfront, 10, vmax=HscaleMax)
    plt.title("front view")
    plt.xlabel('y')
    plt.ylabel('z')
    plt.subplot(1, 3, 2) # back
    plt.imshow(Hback, extent=(-1, 1, -1, 1), aspect=1)
    plt.contourf(grid, grid, Hback, 10, vmax=HscaleMax)
    plt.title("back view")
    plt.xlabel('y')
    plt.ylabel('z')
    plt.subplot(1, 3, 3) # top
    plt.imshow(Htop, extent=(-1, 1, 1, -1), aspect=1)
    plt.contourf(grid, grid, Htop, 10, vmax=HscaleMax)
    plt.title("top view")
    plt.xlabel('y')
    plt.ylabel('x')
    plt.show()

def plot_H_wrapper_interact(it,HfrontAll, HbackAll, HtopAll, grid, HscaleMax):    
    it = int(it)
    plot_H_all(HfrontAll[:,:,it], HbackAll[:,:,it], HtopAll[:,:,it], grid, HscaleMax)