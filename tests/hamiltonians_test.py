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

        evecs_dense = LA.eig(self.H_mat.toarray())[1][:, :self.k]
        overlaps = np.zeros(self.k, dtype=complex)
        for i in range(self.k):
            overlap = np.vdot(self.evecs_num[:, i], evecs_dense[:, i])
            overlaps[i] = overlap
        assert np.allclose(np.abs(overlaps), 1) # check that the eigenvectors are approximately the same, up to a global phase
