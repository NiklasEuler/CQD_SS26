import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
import Comp_Quant_Dynam.hamiltonians as ham
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops


class Test_ladder_operators:

    N = 20

    a_op = ops.a_operator_sparse(N)
    adag_op = ops.adag_operator_sparse(N)
    n_op = ops.n_operator_sparse(N)
    x_op = ops.x_operator_sparse(N)
    p_op = ops.p_operator_sparse(N)

    def test_a_adag_commutation(self):
        commutator = self.a_op @ self.adag_op - self.adag_op @ self.a_op
        identity = sparse.eye(self.N)
        diff = commutator - identity
        assert np.allclose(diff.data[:-1], 0)

    def test_n_identity(self):
        n_from_a_adag = self.adag_op @ self.a_op
        diff = n_from_a_adag - self.n_op
        assert np.allclose(diff.data, 0)

    def test_x_p_commutation(self):
        commutator = self.x_op @ self.p_op - self.p_op @ self.x_op
        identity = 1j * sparse.eye(self.N)
        diff = commutator - identity
        diff_dense = diff.toarray()
        assert np.allclose(diff_dense[:-1, :-1], 0) # the last row and column devite due to the truncation of the operators.

    def test_a_adag_hermiticity(self):
        assert np.allclose(self.a_op.data, self.adag_op.conj().T.data)
    
    def test_n_hermiticity(self):
        assert np.allclose(self.n_op.data, self.n_op.data.conj())
    
    def test_x_hermiticity(self):
        diff = self.x_op - self.x_op.T.conj()
        assert np.allclose(diff.data, 0)
    
    def test_p_hermiticity(self):
        diff = self.p_op - self.p_op.T.conj()
        assert np.allclose(diff.data, 0)

class Test_n_proj_operator_sparse:

    N1 = 5
    N2 = 5
    N_vec = [N1, N2]

    n_proj_a_2 = ops.n_proj_operator_sparse(N_vec, 0, 2).toarray()
    n_proj_b_3 = ops.n_proj_operator_sparse(N_vec, 1, 3).toarray()

    state_2_3 = np.kron(np.eye(1, N1, 2).flatten(), np.eye(1, N2, 3).flatten()) # state |2,3>
    state_1_1 = np.kron(np.eye(1, N1, 1).flatten(), np.eye(1, N2, 1).flatten()) # state |1,1>
    state_2_0 = np.kron(np.eye(1, N1, 2).flatten(), np.eye(1, N2, 0).flatten()) # state |2,0>

    def test_n_proj_operator_sparse_0_2(self):
        state_0_2_proj_a = self.n_proj_a_2 @ self.state_2_3
        state_0_2_proj_b = self.n_proj_b_3 @ self.state_2_3
        assert np.allclose(state_0_2_proj_a, self.state_2_3)
        assert np.allclose(state_0_2_proj_b, self.state_2_3)

    def test_n_proj_operator_sparse_1_3(self):
        state_1_3_proj_a = self.n_proj_a_2 @ self.state_1_1
        state_1_3_proj_b = self.n_proj_b_3 @ self.state_1_1
        assert np.allclose(state_1_3_proj_a, 0)
        assert np.allclose(state_1_3_proj_b, 0)

    def test_n_proj_operator_sparse_2_0(self):
        state_2_0_proj_a = self.n_proj_a_2 @ self.state_2_0
        state_2_0_proj_b = self.n_proj_b_3 @ self.state_2_0
        assert np.allclose(state_2_0_proj_a, self.state_2_0)
        assert np.allclose(state_2_0_proj_b, 0)