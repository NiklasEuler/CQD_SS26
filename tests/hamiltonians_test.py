import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigsh
import Comp_Quant_Dynam.hamiltonians as ham
from Comp_Quant_Dynam.utility import create_xvals


#################### Solution sheet 1 ####################

class Test_HO_eigenstates_exact:

    L = 10
    npoints = 101
    xvals, dx = create_xvals(L, npoints)
    
    def test_HO_ground(self):
        n0 = 0
        expected =  1 / np.pi ** (1 / 4) * np.exp(-self.xvals ** 2 / 2)
        result = ham.HO_eigenstates_exact(n0, self.xvals)
        assert np.allclose(expected, result)

class Test_HO_eigenenergies:
    
    L = 15
    npoints = 401
    acc = 1e-2
    #npoints = 2001
    #acc = 1e-3
    xvals, dx = create_xvals(L, npoints)


    def test_HO_ED(self):
        
        H_pot = ham.HO_potential(self.xvals)
        H_kin = ham.H_kinetic(self.xvals)

        H_mat = H_pot + H_kin

        evals_num, evecs_num = LA.eigh(H_mat)
        evals_exact = ham.HO_eigenenergies_exact(np.arange(evals_num.size))
        assert np.allclose(evals_num[:10], evals_exact[:10], atol=self.acc)

#################### Solution sheet 2 ####################

class Test_HO_sparse_eigenenergies:
    
    L = 15
    npoints = 401
    acc = 1e-2
    k = 10
    xvals, dx = create_xvals(L, npoints)

    H_pot_sparse = ham.HO_potential_sparse(xvals)
    H_kin_sparse = ham.H_kinetic_sparse(xvals)

    H_mat = H_pot_sparse + H_kin_sparse
    evals_num, evecs_num = eigsh(H_mat, k=k, which='SA')

    def test_HO_ED_evals(self):
        
        evals_exact = ham.HO_eigenenergies_exact(np.arange(self.evals_num.size))
        assert np.allclose(self.evals_num, evals_exact, atol=self.acc)

    def test_HO_ED_evecs(self):

        evecs_dense = LA.eigh(self.H_mat.toarray())[1][:, :self.k]
        overlaps = np.zeros(self.k, dtype=complex)
        for i in range(self.k):
            overlap = np.vdot(self.evecs_num[:, i], evecs_dense[:, i])
            overlaps[i] = overlap
        assert np.allclose(np.abs(overlaps), 1) # check that the eigenvectors are approximately the same, up to a global phase


#################### Solution sheet 3 ####################


class Test_potentials:
    L = 2
    npoints = 21
    xvals, dx = create_xvals(L, npoints)
    
    def test_step_potential(self):
        V0 = 5
        expected = np.zeros_like(self.xvals)
        expected[self.xvals >= 0] = V0
        result = ham.step_potential(self.xvals, V0)
        assert np.allclose(expected, result)

    def test_barrier_potential(self):
        V0 = 5
        width = 1
        expected = np.zeros_like(self.xvals)
        expected[np.abs(self.xvals) <= width / 2] = V0
        result = ham.barrier_potential(self.xvals, V0, width)
        assert np.allclose(expected, result)


##################### Exercise sheet 4 ####################


class Test_build_H_coupled_HO_man:
    N1 = 10
    N2 = 10

    def test_build_H_coupled_HO_man_hermitian(self):
        lam = 0.1
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        diff = H - H.conj().T
        assert np.allclose(diff.nnz, 0) # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_coupled_HO_man_non_interacting(self):
        lam = 0
        k = 10
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        evals, evecs = eigsh(H, k=k, which='SA')
        evals_uncoupled = np.add.outer(ham.HO_eigenenergies_exact(np.arange(self.N1)), ham.HO_eigenenergies_exact(np.arange(self.N2))).flatten()
        evals_uncoupled.sort()
        assert np.allclose(evals, evals_uncoupled[:k]) # check that the eigen