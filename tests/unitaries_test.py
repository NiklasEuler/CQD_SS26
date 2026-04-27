import numpy as np
import Comp_Quant_Dynam.unitaries as unit
import Comp_Quant_Dynam.utility as util


#################### Solution sheet 2 ####################


class Test_init_coeffs_eigenbasis:

    H = np.array([[0, 1], [1, 0]]) # simple \sigma_x Hamiltonian
    evals, evecs = np.linalg.eigh(H)

    def test_init_coeffs_eigenbasis_HO_ground(self):
        psi0 = np.array([1, -1]) * 1/np.sqrt(2) # ground state of \sigma_x
        expected = np.array([1, 0]) # in the eigenbasis of \sigma_x, the ground state
        result = unit.init_coeffs_eigenbasis(psi0, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the initial state is approximately the ground state in the eigenbasis, up to a global phase

    def test_init_coeffs_eigenbasis_HO_computational(self):
        psi0 = np.array([0, 1]) # state in the computational basis
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # in the eigenbasis of \sigma_x, this is an equal superposition of the two eigenvectors
        result = unit.init_coeffs_eigenbasis(psi0, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the initial state is approximately the expected state in the eigenbasis, up to a global phase

class Test_t_evol_eigenbasis:

    H = np.array([[0, 1], [1, 0]]) # simple \sigma_x Hamiltonian
    evals, evecs = np.linalg.eigh(H)

    def test_t_evol_eigenbasis_HO_ground(self):
        init_coeffs = np.array([1, 0]) # ground state in the eigenbasis
        t = np.pi / 2
        expected = np.array([1j, -1j]) * 1/np.sqrt(2) # after time evolution, this should be the state in the computational basis corresponding to the second eigenvector
        result = unit.t_evol_eigenbasis(init_coeffs, t, self.evals, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the evolved state

    def test_t_evol_eigenbasis_HO_computational(self):
        init_coeffs = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # equal superposition in the eigenbasis
        t = np.pi / 2
        expected = np.array([1, 0]) # after time evolution, this should be the state in the computational basis.
        result = unit.t_evol_eigenbasis(init_coeffs, t, self.evals, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the evolved state is approximately the expected state in the computational basis, up to a global phase

    