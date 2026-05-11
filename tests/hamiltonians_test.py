import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigsh
import Comp_Quant_Dynam.hamiltonians as ham
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops
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
        tol = 1e-10
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        diff = H - H.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_coupled_HO_man_non_interacting(self):
        lam = 0
        k = 15
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        assert H.nnz == self.N1 * self.N2  # check that the number of non-zero elements in H is equal to N1*N2
        evals, evecs = eigsh(H, k=k+2, which='SA') # compute the lowest k+2 eigenvalues and eigenvectors of H using the sparse eigensolver
        evals_uncoupled = np.add.outer(ham.HO_eigenenergies_exact(np.arange(self.N1)), ham.HO_eigenenergies_exact(np.arange(self.N2))).flatten()
        evals_uncoupled.sort()
        print("evals:", evals)
        print("evals_uncoupled:", evals_uncoupled[:k])
        assert np.allclose(evals[:k], evals_uncoupled[:k])

    def test_build_H_coupled_HO_man_lam_effect(self):
        lam = 0.5
        k = 15
        dim = self.N1 * self.N2
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        evals, evecs = eigsh(H, k=k+2, which='SA')
        print("evals with coupling:", evals)
        evals_coupled = [ham.coupled_HO_eigenenergies_exact(n_cm, n_rel, lam) for n_cm in range(dim) for n_rel in range(dim)]
        evals_coupled.sort()
        assert np.allclose(evals[:k], evals_coupled[:k], atol=1e-3)


class Test_build_H_coupled_HO_improved:
    N1 = 10
    N2 = 10

    def test_build_H_coupled_HO_improved_hermitian(self):
        lam = 0.1
        tol = 1e-10
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        diff = H - H.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_coupled_HO_improved_non_interacting(self):
        lam = 0
        k = 15
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        assert H.nnz == self.N1 * self.N2  # check that the number of non-zero elements in H is equal to N1*N2
        evals, evecs = eigsh(H, k=k+2, which='SA') # compute the lowest k+2 eigenvalues and eigenvectors of H using the sparse eigensolver
        evals_uncoupled = np.add.outer(ham.HO_eigenenergies_exact(np.arange(self.N1)), ham.HO_eigenenergies_exact(np.arange(self.N2))).flatten()
        evals_uncoupled.sort()
        print("evals:", evals)
        print("evals_uncoupled:", evals_uncoupled[:k])
        assert np.allclose(evals[:k], evals_uncoupled[:k])

    def test_build_H_coupled_HO_improved_lam_effect(self):
        lam = 0.5
        k = 15
        dim = self.N1 * self.N2
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        evals, evecs = eigsh(H, k=k+2, which='SA')
        print("evals with coupling:", evals)
        evals_coupled = [ham.coupled_HO_eigenenergies_exact(n_cm, n_rel, lam) for n_cm in range(dim) for n_rel in range(dim)]
        evals_coupled.sort()
        assert np.allclose(evals[:k], evals_coupled[:k], atol=1e-3)

class Test_coupled_HO_E0_exact:

    def test_coupled_HO_E0_exact(self):
        lam = 0.5
        expected = 1 / 2 + np.sqrt(1 + 2 * lam) / 2
        result = ham.coupled_HO_E0_exact(lam)
        assert np.isclose(expected, result)

    def test_coupled_HO_E0_exact_generalized(self):
        lam = 0.35
        expected = ham.coupled_HO_eigenenergies_exact(0, 0, lam)
        result = ham.coupled_HO_E0_exact(lam)
        assert np.isclose(expected, result)

class Test_HO_product_eigenstates:

    N1 = 4
    N2 = 6
    L = 15
    npoints = 401
    xvals, dx = create_xvals(L, npoints)
    eigenstates = ham.HO_product_eigenstates(N1, N2, xvals) # normalize the eigenstates by the grid spacing to ensure that they are properly normalized in the continuum limit

    def test_HO_product_eigenstates_shape(self):

        assert self.eigenstates.shape == (self.N1 * self.N2, self.npoints, self.npoints)

    def test_HO_product_eigenstates_symmetry(self):
        assert np.allclose(self.eigenstates[0], self.eigenstates[0].T)

    def test_HO_product_eigenstates_orthonormality(self):
        overlaps = np.zeros((self.N1 * self.N2, self.N1 * self.N2), dtype=complex)
        for i in range(self.N1 * self.N2):
            for j in range(self.N1 * self.N2):
                overlap = np.vdot(self.eigenstates[i].flatten(), self.eigenstates[j].flatten())
                overlaps[i, j] = np.real(overlap) * self.dx**2  # divide by dx^2 to account for the normalization of the eigenstates in the continuum limit
        assert np.allclose(overlaps, np.eye(self.N1 * self.N2))

    def test_HO_product_eigenstates_example(self):
        N1 = N2 = 8
        eigenstates = ham.HO_product_eigenstates(N1, N2, self.xvals)
        assert np.allclose(eigenstates[9], np.outer(ham.HO_eigenstates_exact(1, self.xvals), ham.HO_eigenstates_exact(1, self.xvals))) 

class Test_class_traj:
    N1 = N2 = 20
    local_dims = [N1, N2]
    x1_op = ops.n_party_op_sparse(local_dims, 0, ops.x_operator_sparse(N1))
    x2_op = ops.n_party_op_sparse(local_dims, 1, ops.x_operator_sparse(N2))
    p1_op = ops.n_party_op_sparse(local_dims, 0, ops.p_operator_sparse(N1))
    p2_op = ops.n_party_op_sparse(local_dims, 1, ops.p_operator_sparse(N2))

    def test_traj_comparison_to_numerics(self):
        alpha = 0.1
        state_1 = util.create_coherent_state(self.N1, alpha)
        state_2 = np.eye(1, self.N2, 4).flatten() # ground state of the second oscillator
        state = np.kron(state_1, state_2) # initial state is the product of the coherent state and the ground state
        x10 = util.expectation_value(state, self.x1_op)
        x20 = util.expectation_value(state, self.x2_op)
        p10 = util.expectation_value(state, self.p1_op)
        p20 = util.expectation_value(state, self.p2_op)

        ini = np.array([x10, x20, p10, p20])
        print("Initial conditions:", ini)
        lam = 0.1

        tvec = util.create_tvecs(100, 0.1)

        x1_class, x2_class = ham.class_traj(lam, ini, tvec)
        x1_num, x2_num = np.zeros_like(tvec), np.zeros_like(tvec)

        H_mat = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam).toarray()
        evals, evecs = LA.eigh(H_mat)
           

        eigen_coeffs = unitaries.init_coeffs_eigenbasis(state, evecs) 
        for i, t in enumerate(tvec):
            state_t = unitaries.t_evol_eigenbasis(eigen_coeffs, t, evals, evecs)
            x1_num[i] = util.expectation_value(state_t, self.x1_op)
            x2_num[i] = util.expectation_value(state_t, self.x2_op)

        assert np.allclose(x1_class, x1_num, atol=1e-8)
        assert np.allclose(x2_class, x2_num, atol=1e-8)


class Test_coupled_HO_potential:

    L = 10
    npoints = 101
    lam = 0.5

    xvals, dx = util.create_xvals(L, npoints)
    yvals, dy = util.create_xvals(L, npoints)

    X, Y = np.meshgrid(xvals, yvals)

    H_pot = ham.coupled_HO_potential(X, Y, lam)

    #def test_coupled_HO_potential(self):
    #    expected = ham.HO_potential(self.X) + ham.HO_potential(self.Y) + self.lam / 2 * (self.X - self.Y) ** 2
    #    assert np.allclose(expected, self.H_pot)

    def test_coupled_HO_potential_symmetry(self):
        assert np.allclose(self.H_pot, self.H_pot.T)

    def test_coupled_HO_potential_diagonal(self):
        expected_no_coupling = 1/2 * self.X**2 + 1/2 * self.Y**2
        assert np.allclose(self.H_pot.diagonal(), expected_no_coupling.diagonal()) # the coupling term should not contribute to the diagonal elements of the potential

