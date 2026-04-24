import numpy as np   # standard numerics library
import math

def example_func(x):
    """
    Example function to demonstrate the repository structure.
    Returns the ground state wavefunction of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    return 1 / np.pi ** (1 / 4) * np.exp(-x ** 2 / 2)


#################### Solution sheet 2 ####################


def create_xvals(L, npoints):
    """
    Creates a grid of 'npoints' evenly spaced values between -L/2 and L/2.
    """
    xvals = np.linspace(-L / 2, L / 2, npoints)
    dx = xvals[1] - xvals[0]
    return xvals, dx