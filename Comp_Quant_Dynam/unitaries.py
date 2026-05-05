import numpy as np   # standard numerics library


#################### Solution sheet 2 ####################


def t_evol_eigenbasis(init_coeffs, t, evals, evecs):
    """
    Returns the state in the computational basis at time 't' given the initial state coefficients 'init_coeffs'
    in the eigenbasis for a Hamiltonian with eigenvalues 'evals' and eigenvectors 'evecs'.
    """

    phase_factors = np.exp(-1j * evals * t) # compute the phase factors for each eigenstate
    evolved_eigen_coeffs = init_coeffs * phase_factors
    return evecs @ evolved_eigen_coeffs

def init_coeffs_eigenbasis(psi0, evecs):
    """
    Returns the coefficients of the initial state 'psi0' in the eigenbasis given by 'evecs'.
    """

    return np.conjugate(evecs.T) @ psi0


#################### Solution sheet 3 ####################


def t_evol_split_step_fourier(psi0, V_func, tvec, xvals):
    """
    Returns the time evolution of the state 'psi0' under the potential 'V_func' using the split-step Fourier method for time evolution.
    V_func is a function that takes time 't' as input and returns the potential at that time.
    The time vector is given by 'tvec', and the position and momentum grids are given by 'x' and 'k', respectively.
    """

    npoints = len(xvals) # number of grid points
    dx = xvals[1] - xvals[0] # grid spacing
    kvals = 2 * np.pi * np.fft.fftfreq(npoints, d=dx)
    tsteps = len(tvec) - 1 # number of time steps
    dt = tvec[1] - tvec[0] # time step size

    # momentum grid corresponding to the position grid
    
    # container for storing the result
    psit = np.zeros((len(tvec), npoints), dtype=complex)

    # store initial value
    psit[0] = psi0

    for i in range(tsteps):
        
        # apply potential
        psit[i + 1, :] = np.exp(-1j * dt * V_func(tvec[i])) * psit[i, :]
        # go to Fourier space
        psit[i + 1, :] = np.fft.fft(psit[i + 1, :])
        # apply kinetic part
        psit[i + 1, :] = np.exp(-1j * dt * kvals**2 / 2) * psit[i + 1, :]
        # go back to real space
        psit[i + 1, :] = np.fft.ifft(psit[i + 1, :])
        # store the result
        psit[i + 1] = psit[i + 1, :]

    return psit
