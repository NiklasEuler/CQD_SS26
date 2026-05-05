import numpy as np   # standard numerics library
import math
from scipy.special import hermite as herm
import scipy.sparse as sparse # routines for sparse matrices

from Comp_Quant_Dynam.utility import state2idx, idx2state

#################### Solution sheet 1 ####################


def HO_eigenstates_exact(n,x):
    """
    Returns the n-th eigenstate of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    normalization = 1 / np.sqrt(2 ** n * math.factorial(n) * np.sqrt(np.pi)) 
    return normalization * herm(n)(x) * np.exp(-x ** 2 / 2)


def HO_eigenenergies_exact(n):
    """
    Returns the n-th eigenenergy of the quantum harmonic oscillator in numerical units.
    """

    return n + 0.5

def H_kinetic(x):
    """
    Returns the kinetic energy operator of the quantum harmonic oscillator in the position basis for a grid 'x'.
    The kinetic energy operator is represented as a finite difference matrix approximating the second derivative, which is given by the formula:
    T = - (ħ^2 / 2m) * d^2/dx^2, or in numerical units, T = -0.5 * d^2/dx^2. The second derivative can be approximated using the central difference formula:
    d^2ψ/dx^2 ≈ (ψ(x + dx) - 2ψ(x) + ψ(x - dx)) / (dx^2).
    """

    n_points = len(x) # number of grid points
    dx = x[1] - x[0] # grid spacing

    main_diag = np.diag(np.ones(n_points))
    off_diag = -0.5 * np.diag(np.ones(n_points - 1), k=1)
    return (main_diag + off_diag + off_diag.T) / (dx * dx)

def HO_potential(x):
    """
    Returns the potential energy operator of the quantum harmonic oscillator in the position basis for a grid 'x'.
    The potential energy operator is represented as a diagonal matrix with elements given by V(x) = 0.5 * x^2.
    """
    
    return 0.5 * np.diag(x ** 2)


#################### Solution sheet 2 ####################


def H_kinetic_sparse(x):
    """
    Returns the kinetic energy operator of the quantum harmonic oscillator in the position basis for a grid 'x' as a sparse matrix.
    The kinetic energy operator is represented as a finite difference matrix approximating the second derivative, which is given by the formula:
    T = - (ħ^2 / 2m) * d^2/dx^2, or in numerical units, T = -0.5 * d^2/dx^2. The second derivative can be approximated using the central difference formula:
    d^2ψ/dx^2 ≈ (ψ(x + dx) - 2ψ(x) + ψ(x - dx)) / (dx^2).
    """

    n_points = len(x) # number of grid points
    dx = x[1] - x[0] # grid spacing

    main_diag = np.ones(n_points)
    off_diag = -0.5 * np.ones(n_points - 1)
    H_kin = sparse.diags_array(
        [main_diag, off_diag, off_diag],
        offsets = [0, 1, -1],
    ) / (dx * dx)
    return H_kin

def HO_potential_sparse(x):
    """
    Returns the potential energy operator of the quantum harmonic oscillator in the position basis for a grid 'x' as a sparse matrix.
    The potential energy operator is represented as a diagonal matrix with elements given by V(x) = 0.5 * x^2.
    """
    
    return sparse.diags_array(0.5 * x ** 2)


#################### Solution sheet 3 ####################


def step_potential(x, V0):
    """
    Returns the potential energy of a step potential of step height 'V0' in the position basis for a grid 'x' as a diagonal matrix.
    The step potential is defined as V(x) = 0 for x < 0 and V(x) = V0 for x >= 0.
    """
    potential = V0 * (1 + np.sign(x)) / 2
    return potential

def barrier_potential(x, V0, width):
    mask = np.heaviside(x + width/2, 1) - np.heaviside(x - width/2, 1)
    return V0 * mask


##################### Exercise sheet 4 ####################


def build_H_coupled_HO_man(N1, N2, lam):
    """
    Manually builds the Hamiltonian matrix for two coupled harmonic oscillators in the number basis, where `N1` and `N2` are the maximum occupation numbers for the two oscillators, and `lam` is the coupling strength.
    The Hamiltonian is given by:
    H = H_1 + H_2 + V = 1 / (2m) * (p_1^2 + p_2^2) + k / 2 * (x_1^2 + x_2^2) + lam / 2 * (x_1 - x_2)^2
    This function is inteded for testing purposes, and better implemented using the ladder operators.
    """
    # build a vector of data-index pairs for the non-zero matrix elements
    values = np.array([])
    row = np.array([])
    col = np.array([])
    for i in range(N1*N2):
        state = idx2state(N1,N2,i)
        # diagonal term (k=m=hbar=1)
        values = np.append(values,(1+lam/2)*(1+state[0]+state[1]))
        row = np.append(row,i)
        col = np.append(col,i)
        # off-diagonal elements of a column (8 terms in the perturbing Hamiltonian)
        # a1^dag a1^dag
        stateCoupleTo = [state[0] + 2, state[1] + 0]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[0]+1)*(state[0]+2)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a1
        stateCoupleTo = [state[0] - 2, state[1] + 0]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[0])*(state[0]-1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a2^dag a2^dag
        stateCoupleTo = [state[0] + 0, state[1] + 2]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[1]+1)*(state[1]+2)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a2 a2
        stateCoupleTo = [state[0] + 0, state[1] - 2]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[1])*(state[1]-1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1^dag a2^dag
        stateCoupleTo = [state[0] + 1, state[1] + 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0]+1)*(state[1]+1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1^dag a2
        stateCoupleTo = [state[0] + 1, state[1] - 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0]+1)*(state[1])))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a2^dag
        stateCoupleTo = [state[0] - 1, state[1] + 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0])*(state[1]+1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a2
        stateCoupleTo = [state[0] - 1, state[1] -1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0])*(state[1])))
            row = np.append(row,indTo)
            col = np.append(col,i)

    return sparse.coo_matrix((values, (row, col)), shape=(N1*N2, N1*N2))
