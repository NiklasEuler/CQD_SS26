import numpy as np
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.operators as ops


class Test_example_function:

    def test_example_func_zero(self):
        x = 0
        expected =  1 / np.pi ** (1 / 4)
        result = util.example_func(x)
        assert np.allclose(expected, result)

    def test_example_func_symmetry(self):
        x = np.array([-1, 1])
        result = util.example_func(x)
        assert np.allclose(result[0], result[1])


#################### Solution sheet 2 ####################


class Test_create_xvals:

    L = 10
    npoints = 101
    
    def test_create_xvals_length(self):
        
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert len(xvals) == self.npoints

    def test_create_xvals_range(self):

        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[0], -self.L/2)
        assert np.isclose(xvals[-1], self.L/2)

    def test_create_xvals_spacing(self):
        xvals, dx = util.create_xvals(self.L, self.npoints)
        expected_dx = self.L / (self.npoints - 1)
        assert np.isclose(dx, expected_dx)

    def test_create_xvals_zero_centered(self):
        # only works if npoints is odd
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[self.npoints // 2], 0) # check that the middle point is approximately zero


##################### Solution sheet 3 ####################


class Test_FT_iFT:

    L = 10
    npoints = 101
    xvals, dx = util.create_xvals(L, npoints, endpoint=False)
    kvals = np.fft.fftfreq(npoints, d=dx) * 2 * np.pi

    def test_FT_iFT_identity(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        phi = util.FT(psi, self.xvals, self.kvals)
        psi_reconstructed = util.iFT(phi, self.xvals, self.kvals)
        assert np.allclose(psi, psi_reconstructed)


    def test_FT_assertion(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        with np.testing.assert_raises(AssertionError):
            util.FT(psi, self.xvals, self.kvals[:-1]) # mismatch in length of k

    def test_iFT_assertion(self):
        phi = util.FT(util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1), self.xvals, self.kvals)
        with np.testing.assert_raises(AssertionError):
            util.iFT(phi, self.xvals[:-1], self.kvals) # mismatch in length of x

class Test_gaussian_wave_packet:

    L = 20
    npoints = 201
    xvals, dx = util.create_xvals(L, npoints)

    def test_gaussian_wave_packet_normalization(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=1)
        norm = np.sum(np.abs(psi)**2) * self.dx
        assert np.isclose(norm, 1)

    def test_gaussian_wave_packet_symmetry(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=0)
        assert np.allclose(psi[60], psi[80]) # check that the wave packet is symmetric around x0=-3

class Test_create_tvecs:

    def test_create_tvecs_length(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert len(tvec) == tsteps + 1

    def test_create_tvecs_values(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert np.isclose(tvec[0], 0)
        assert np.isclose(tvec[-1], tsteps * dt)

class Test_idx2state_state2idx:

    N1 = 3
    N2 = 4

    def test_idx2state_state2idx_consistency(self):
        idx_recon = []
        for i in range(self.N1 * self.N2):
            state = util.idx2state(self.N1, self.N2, i)
            idx_recon.append(util.state2idx(self.N1, self.N2, state))
        assert np.array_equal(np.arange(self.N1 * self.N2), idx_recon)

    def test_idx2state_state2idx_out_of_bounds(self):
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, -1) # negative index
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, self.N1 * self.N2) # index equal to dimension
        idx = util.state2idx(self.N1, self.N2, [self.N1, 0]) # n1 out of bounds
        assert idx == -1
        idx = util.state2idx(self.N1, self.N2, [0, self.N2]) # n2 out of bounds
        assert idx == -1

    def test_idx2state_specific_cases(self):
        # test some specific cases for idx2state and state2idx
        i = 0
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [0, 0]
        i = 5
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [1, 1]
        i = 11
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [2, 3]


##################### Solution sheet 4 ####################


class Test_create_coherent_state:

    N = 100

    def test_create_coherent_state_normalization(self):
        alpha = 1 + 1j
        state = util.create_coherent_state(self.N, alpha)
        norm = np.sum(np.abs(state)**2)
        print(f"Norm of the coherent state: {norm}")
        assert np.isclose(norm, 1)

    def test_create_coherent_state_alpha_zero(self):
        alpha = 0
        state = util.create_coherent_state(self.N, alpha)
        expected = np.zeros(self.N)
        expected[0] = 1
        assert np.allclose(state, expected)

    def test_adag_operator_sparse_consistency(self):
        alpha = 1
        a_op = ops.a_operator_sparse(self.N).toarray()
        init_state = util.create_coherent_state(self.N, alpha)
        applied_a_op = a_op @ init_state
        applied_a_op /= alpha # should be equal to the original state
        assert np.allclose(applied_a_op, init_state)